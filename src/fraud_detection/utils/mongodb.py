from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from fraud_detection.constants.constants import (
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
    REPO_ROOT,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 10_000
TIMESTAMP_COLUMNS = ("createdAt", "trans_date", "updatedAt")

MONGO_PROJECTION = {
    "member_id": 1,
    "draw_id": 1,
    "bets": 1,
    "win_points": 1,
    "total_bet_amount": 1,
    "session_id": 1,
    "ccs_id": 1,
    "createdAt": 1,
    "updatedAt": 1,
    "trans_date": 1,
}


def get_mongo_collection(uri_env_var: str, db_env_var: str, collection_env_var: str):
    try:
        load_dotenv(REPO_ROOT / ".env")
        from pymongo import MongoClient

        uri = os.getenv(uri_env_var)
        database = os.getenv(db_env_var)
        collection_name = os.getenv(collection_env_var)
        if not uri or not database or not collection_name:
            raise ValueError(
                f"Missing MongoDB env vars: {uri_env_var}, {db_env_var}, {collection_env_var}. "
                "Check your .env file."
            )
        client = MongoClient(uri, serverSelectionTimeoutMS=15000)
        return client, client[database][collection_name]
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def pull_full_collection(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    query_filter: dict | None = None,
) -> pd.DataFrame:
    try:
        logger.info("Connecting to MongoDB collection via env vars")
        client, collection = get_mongo_collection(uri_env_var, db_env_var, collection_env_var)
        try:
            cursor = collection.find(
                query_filter or {},
                MONGO_PROJECTION,
                no_cursor_timeout=True,
            ).batch_size(BATCH_SIZE)
            docs: list[dict] = []
            for index, doc in enumerate(cursor, start=1):
                docs.append(doc)
                if index % BATCH_SIZE == 0:
                    logger.info("Pulled %d documents from MongoDB", index)
            logger.info("Pulled %d documents from MongoDB", len(docs))
        finally:
            client.close()
        if not docs:
            raise ValueError("MongoDB query returned 0 documents.")
        return pd.DataFrame(docs)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def _update_timestamp_stats(
    stats: dict[str, dict[str, Any]],
    batch_df: pd.DataFrame,
) -> None:
    for column in TIMESTAMP_COLUMNS:
        if column not in batch_df.columns:
            continue
        parsed = pd.to_datetime(batch_df[column], errors="coerce", utc=True)
        valid = parsed.dropna()
        if valid.empty:
            continue
        batch_min = valid.min()
        batch_max = valid.max()
        current = stats[column]
        if current["min"] is None or batch_min < current["min"]:
            current["min"] = batch_min
        if current["max"] is None or batch_max > current["max"]:
            current["max"] = batch_max


def _finalize_date_range(stats: dict[str, dict[str, Any]]) -> dict[str, str]:
    for column in TIMESTAMP_COLUMNS:
        current = stats[column]
        if current["min"] is not None and current["max"] is not None:
            return {
                "from": str(current["min"]),
                "to": str(current["max"]),
            }
    return {}


def _serialize_bets(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, default=str)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return json.dumps(value, default=str)


def _normalize_batch_for_parquet(batch_docs: list[dict[str, Any]], projected_columns: list[str]) -> pd.DataFrame:
    batch_df = pd.DataFrame(batch_docs).reindex(columns=projected_columns)

    if "member_id" in batch_df.columns:
        batch_df["member_id"] = batch_df["member_id"].astype("string")
    if "ccs_id" in batch_df.columns:
        batch_df["ccs_id"] = batch_df["ccs_id"].astype("string")
    if "draw_id" in batch_df.columns:
        batch_df["draw_id"] = pd.to_numeric(batch_df["draw_id"], errors="coerce").astype("Int64")
    if "session_id" in batch_df.columns:
        batch_df["session_id"] = pd.to_numeric(batch_df["session_id"], errors="coerce").astype("Int64")
    if "win_points" in batch_df.columns:
        batch_df["win_points"] = pd.to_numeric(batch_df["win_points"], errors="coerce").astype("float64")
    if "total_bet_amount" in batch_df.columns:
        batch_df["total_bet_amount"] = pd.to_numeric(batch_df["total_bet_amount"], errors="coerce").astype("float64")
    if "bets" in batch_df.columns:
        batch_df["bets"] = batch_df["bets"].apply(_serialize_bets).astype("string")
    for column in TIMESTAMP_COLUMNS:
        if column in batch_df.columns:
            batch_df[column] = pd.to_datetime(batch_df[column], errors="coerce")

    return batch_df


def stream_collection_to_parquet(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    output_paths: list[Path],
    query_filter: dict | None = None,
) -> dict[str, Any]:
    projected_columns = list(MONGO_PROJECTION.keys())
    target_paths = list(dict.fromkeys(output_paths))
    writers: dict[Path, pq.ParquetWriter] = {}
    row_count = 0
    member_ids: set[Any] = set()
    timestamp_stats = {
        column: {"min": None, "max": None}
        for column in TIMESTAMP_COLUMNS
    }
    cursor = None

    def flush_batch(batch_docs: list[dict[str, Any]]) -> None:
        nonlocal row_count
        if not batch_docs:
            return

        batch_df = _normalize_batch_for_parquet(batch_docs, projected_columns)
        row_count += len(batch_df)

        if "member_id" in batch_df.columns:
            member_ids.update(batch_df["member_id"].dropna().tolist())

        _update_timestamp_stats(timestamp_stats, batch_df)

        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        for path in target_paths:
            if path not in writers:
                path.parent.mkdir(parents=True, exist_ok=True)
                writers[path] = pq.ParquetWriter(str(path), table.schema)
            writers[path].write_table(table)

    try:
        logger.info("Connecting to MongoDB collection via env vars")
        client, collection = get_mongo_collection(uri_env_var, db_env_var, collection_env_var)
        try:
            cursor = collection.find(
                query_filter or {},
                MONGO_PROJECTION,
                no_cursor_timeout=True,
            ).batch_size(BATCH_SIZE)

            batch_docs: list[dict[str, Any]] = []
            for index, doc in enumerate(cursor, start=1):
                batch_docs.append(doc)
                if len(batch_docs) >= BATCH_SIZE:
                    flush_batch(batch_docs)
                    logger.info("Streamed %d documents from MongoDB", index)
                    batch_docs = []

            flush_batch(batch_docs)
            logger.info("Streamed %d documents from MongoDB", row_count)
        finally:
            if cursor is not None:
                cursor.close()
            for writer in writers.values():
                writer.close()
            client.close()

        if row_count == 0:
            raise ValueError("MongoDB query returned 0 documents.")

        return {
            "row_count": row_count,
            "member_count": len(member_ids),
            "date_range": _finalize_date_range(timestamp_stats),
        }
    except Exception as e:
        for path in target_paths:
            if path.exists():
                path.unlink()
        raise FraudDetectionException(e, sys) from e
