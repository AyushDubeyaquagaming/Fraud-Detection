from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
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
MEMBER_LIST_CHUNK_SIZE = 10_000
TIMESTAMP_COLUMNS = ("createdAt", "trans_date", "updatedAt")

MONGO_PROJECTION = {
    "_id": 0,
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


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_SERVING_CLIENTS: dict[tuple[str, str, str], Any] = {}
_SERVING_CLIENTS_LOCK = RLock()


def _resolve_mongo_connection_settings(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
) -> tuple[str, str, str]:
    load_dotenv(REPO_ROOT / ".env")
    uri = os.getenv(uri_env_var)
    database = os.getenv(db_env_var)
    collection_name = os.getenv(collection_env_var)
    if not uri or not database or not collection_name:
        raise ValueError(
            f"Missing MongoDB env vars: {uri_env_var}, {db_env_var}, {collection_env_var}. "
            "Check your .env file."
        )
    return uri, database, collection_name


def get_mongo_collection(uri_env_var: str, db_env_var: str, collection_env_var: str):
    try:
        from pymongo import MongoClient

        uri, database, collection_name = _resolve_mongo_connection_settings(
            uri_env_var,
            db_env_var,
            collection_env_var,
        )
        client = MongoClient(uri, serverSelectionTimeoutMS=15000)
        return client, client[database][collection_name]
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def get_serving_mongo_collection(
    uri_env_var: str = ENV_MONGODB_URI,
    db_env_var: str = ENV_MONGODB_DATABASE,
    collection_env_var: str = ENV_MONGODB_COLLECTION,
):
    """Return a process-level Mongo collection for serving paths.

    PyMongo is designed around long-lived clients with internal pooling.
    Serving requests should reuse one client per process rather than opening
    and closing a new client on every call.
    """
    try:
        from pymongo import MongoClient

        uri, database, collection_name = _resolve_mongo_connection_settings(
            uri_env_var,
            db_env_var,
            collection_env_var,
        )
        cache_key = (uri, database, collection_name)
        with _SERVING_CLIENTS_LOCK:
            client = _SERVING_CLIENTS.get(cache_key)
            if client is None:
                client = MongoClient(uri, serverSelectionTimeoutMS=15000)
                _SERVING_CLIENTS[cache_key] = client
        return client[database][collection_name]
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


# ---------------------------------------------------------------------------
# Batch normalisation helpers (shared by all execution paths)
# ---------------------------------------------------------------------------

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


def _normalize_batch_for_parquet(
    batch_docs: list[dict[str, Any]],
    projected_columns: list[str],
) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Strategy query builders (private)
# ---------------------------------------------------------------------------

def _build_date_window_queries(strategy_params: dict[str, Any]) -> list[dict]:
    """Build a date-bounded query using trans_date (or a configured timestamp field)."""
    timestamp_field = strategy_params.get("timestamp_field", "trans_date")
    start_date = strategy_params.get("start_date")
    end_date = strategy_params.get("end_date")
    lookback_days = strategy_params.get("lookback_days")

    if start_date is not None and end_date is not None:
        start_dt: datetime = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
        end_dt: datetime = pd.Timestamp(end_date, tz="UTC").to_pydatetime()
    elif lookback_days is not None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(lookback_days))
    else:
        raise FraudDetectionException(
            ValueError(
                "date_window strategy requires either (start_date + end_date) or "
                "lookback_days in strategy_params."
            ),
            sys,
        )

    logger.info(
        "date_window query: %s in [%s, %s)",
        timestamp_field,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )
    return [{timestamp_field: {"$gte": start_dt, "$lt": end_dt}}]


def _resolve_member_ids(
    strategy_params: dict[str, Any],
    fraud_csv_path: str | Path | None = None,
) -> list[str]:
    """Resolve the list of member IDs from the configured source."""
    source = strategy_params.get("member_ids_source", "inline")

    if source == "inline":
        ids = strategy_params.get("member_ids", [])
        if not ids:
            raise FraudDetectionException(
                ValueError(
                    "member_list strategy with source='inline' requires a non-empty "
                    "member_ids list in strategy_params."
                ),
                sys,
            )
        return [str(m) for m in ids]

    elif source == "file":
        file_path = strategy_params.get("member_ids_file")
        col = strategy_params.get("member_ids_column", "member_id")
        if not file_path:
            raise FraudDetectionException(
                ValueError(
                    "member_list strategy with source='file' requires "
                    "member_ids_file in strategy_params."
                ),
                sys,
            )
        df = pd.read_csv(Path(file_path))
        if col not in df.columns:
            raise FraudDetectionException(
                ValueError(
                    f"Column '{col}' not found in {file_path}. "
                    f"Available columns: {df.columns.tolist()}"
                ),
                sys,
            )
        return df[col].dropna().astype(str).unique().tolist()

    elif source == "fraud_csv":
        csv_path = fraud_csv_path or strategy_params.get("member_ids_file")
        col = strategy_params.get("member_ids_column", "member_id")
        if not csv_path:
            raise FraudDetectionException(
                ValueError(
                    "member_list strategy with source='fraud_csv' requires either "
                    "fraud_csv_path to be passed or member_ids_file set in strategy_params."
                ),
                sys,
            )
        df = pd.read_csv(Path(csv_path))
        if col not in df.columns:
            raise FraudDetectionException(
                ValueError(
                    f"Column '{col}' not found in {csv_path}. "
                    f"Available columns: {df.columns.tolist()}"
                ),
                sys,
            )
        return df[col].dropna().astype(str).unique().tolist()

    else:
        raise FraudDetectionException(
            ValueError(
                f"Unknown member_ids_source '{source}'. "
                "Must be one of: inline, file, fraud_csv."
            ),
            sys,
        )


def _build_member_list_queries(
    strategy_params: dict[str, Any],
    fraud_csv_path: str | Path | None = None,
) -> list[dict]:
    """Build chunked $in queries for a member list (deterministic 10k-ID chunks)."""
    member_ids = _resolve_member_ids(strategy_params, fraud_csv_path)
    chunks = [
        member_ids[i : i + MEMBER_LIST_CHUNK_SIZE]
        for i in range(0, len(member_ids), MEMBER_LIST_CHUNK_SIZE)
    ]
    logger.info(
        "member_list: %d total IDs → %d query batch(es) of up to %d each",
        len(member_ids),
        len(chunks),
        MEMBER_LIST_CHUNK_SIZE,
    )
    return [{"member_id": {"$in": chunk}} for chunk in chunks]


def _validate_full_pull_confirmed(strategy_params: dict[str, Any]) -> None:
    """Raise immediately if full_collection pull is not explicitly confirmed."""
    if not strategy_params.get("confirm_full_pull", False):
        raise FraudDetectionException(
            ValueError(
                "full_collection strategy requires confirm_full_pull=true in strategy_params. "
                "This guard prevents accidental full-collection pulls on the 40M+ document "
                "collection. Set confirm_full_pull: true only when you genuinely intend a "
                "full pull (e.g. one-time baseline extraction)."
            ),
            sys,
        )


# ---------------------------------------------------------------------------
# Public strategy API
# ---------------------------------------------------------------------------

def build_query_batches_from_strategy(
    strategy: str,
    strategy_params: dict[str, Any],
    fraud_csv_path: str | Path | None = None,
) -> list[dict]:
    """Return one or more MongoDB query filter dicts for the given strategy.

    Strategies
    ----------
    date_window
        Default. Bounds by trans_date (or configured timestamp_field).
        Requires lookback_days OR (start_date + end_date) in strategy_params.
    member_list
        Targeted extraction for a known cohort.
        Requires member_ids_source ('inline' | 'file' | 'fraud_csv') in strategy_params.
        Splits into deterministic 10,000-ID chunks.
    full_collection
        Escape hatch for one-time baseline pulls.
        Requires confirm_full_pull=true in strategy_params — raises otherwise.
    """
    if strategy == "date_window":
        return _build_date_window_queries(strategy_params)

    elif strategy == "member_list":
        return _build_member_list_queries(strategy_params, fraud_csv_path)

    elif strategy == "full_collection":
        _validate_full_pull_confirmed(strategy_params)
        logger.warning(
            "full_collection strategy confirmed — pulling ALL documents. "
            "This may take multiple hours on a 40M+ document collection."
        )
        return [{}]

    else:
        raise FraudDetectionException(
            ValueError(
                f"Unknown ingestion strategy '{strategy}'. "
                "Must be one of: date_window, member_list, full_collection."
            ),
            sys,
        )


def stream_query_batches_to_parquet(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    output_paths: list[Path],
    query_filters: list[dict],
) -> dict[str, Any]:
    """Execute multiple query filters and stream all results to shared parquet file(s).

    All query batches are appended to the same parquet writer(s), so member_list
    chunking and multi-batch strategies produce a single coherent output file.

    Returns
    -------
    dict with row_count, member_count, date_range, query_count, strategy_used
    """
    projected_columns = list(MONGO_PROJECTION.keys())
    target_paths = list(dict.fromkeys(output_paths))
    writers: dict[Path, pq.ParquetWriter] = {}
    row_count = 0
    member_ids: set[Any] = set()
    timestamp_stats = {
        column: {"min": None, "max": None} for column in TIMESTAMP_COLUMNS
    }

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
        logger.info(
            "stream_query_batches_to_parquet: connecting — %d query batch(es) to execute",
            len(query_filters),
        )
        client, collection = get_mongo_collection(uri_env_var, db_env_var, collection_env_var)
        try:
            for batch_idx, query_filter in enumerate(query_filters, start=1):
                logger.info(
                    "Query batch %d/%d: %s", batch_idx, len(query_filters), query_filter
                )
                cursor = collection.find(
                    query_filter,
                    MONGO_PROJECTION,
                    no_cursor_timeout=True,
                ).batch_size(BATCH_SIZE)
                try:
                    batch_docs: list[dict[str, Any]] = []
                    docs_in_query = 0
                    for doc in cursor:
                        batch_docs.append(doc)
                        docs_in_query += 1
                        if len(batch_docs) >= BATCH_SIZE:
                            flush_batch(batch_docs)
                            logger.info(
                                "  batch %d/%d: streamed %d docs (running total: %d)",
                                batch_idx,
                                len(query_filters),
                                docs_in_query,
                                row_count,
                            )
                            batch_docs = []
                    flush_batch(batch_docs)
                    logger.info(
                        "  batch %d/%d complete — %d docs, running total: %d",
                        batch_idx,
                        len(query_filters),
                        docs_in_query,
                        row_count,
                    )
                finally:
                    cursor.close()
        finally:
            for writer in writers.values():
                writer.close()
            client.close()

        if row_count == 0:
            raise ValueError("All query batches returned 0 documents combined.")

        logger.info(
            "stream_query_batches_to_parquet: complete — %d rows, %d members, %d query batch(es)",
            row_count,
            len(member_ids),
            len(query_filters),
        )
        return {
            "row_count": row_count,
            "member_count": len(member_ids),
            "date_range": _finalize_date_range(timestamp_stats),
            "query_count": len(query_filters),
        }
    except Exception as e:
        for path in target_paths:
            if path.exists():
                path.unlink()
        raise FraudDetectionException(e, sys) from e


def pull_query_batches_to_dataframe(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    query_filters: list[dict],
) -> pd.DataFrame:
    """Execute multiple query filters and return a single combined DataFrame.

    Intended for batch scoring where an in-memory DataFrame is needed rather
    than a parquet file on disk.
    """
    try:
        logger.info(
            "pull_query_batches_to_dataframe: connecting — %d query batch(es) to execute",
            len(query_filters),
        )
        client, collection = get_mongo_collection(uri_env_var, db_env_var, collection_env_var)
        all_docs: list[dict[str, Any]] = []
        try:
            for batch_idx, query_filter in enumerate(query_filters, start=1):
                logger.info(
                    "Query batch %d/%d: %s", batch_idx, len(query_filters), query_filter
                )
                cursor = collection.find(
                    query_filter,
                    MONGO_PROJECTION,
                    no_cursor_timeout=True,
                ).batch_size(BATCH_SIZE)
                try:
                    docs_in_query = 0
                    for doc in cursor:
                        all_docs.append(doc)
                        docs_in_query += 1
                        if docs_in_query % BATCH_SIZE == 0:
                            logger.info(
                                "  batch %d/%d: pulled %d docs (total so far: %d)",
                                batch_idx,
                                len(query_filters),
                                docs_in_query,
                                len(all_docs),
                            )
                    logger.info(
                        "  batch %d/%d complete — %d docs", batch_idx, len(query_filters), docs_in_query
                    )
                finally:
                    cursor.close()
        finally:
            client.close()

        if not all_docs:
            raise ValueError("All query batches returned 0 documents combined.")

        projected_columns = list(MONGO_PROJECTION.keys())
        df = _normalize_batch_for_parquet(all_docs, projected_columns)
        member_count = int(df["member_id"].nunique()) if "member_id" in df.columns else 0
        logger.info(
            "pull_query_batches_to_dataframe: complete — %d rows, %d members, %d query batch(es)",
            len(df),
            member_count,
            len(query_filters),
        )
        return df
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


# ---------------------------------------------------------------------------
# Legacy single-query helpers (preserved for existing callers)
# ---------------------------------------------------------------------------

def pull_full_collection(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    query_filter: dict | None = None,
) -> pd.DataFrame:
    """Pull the entire collection (or a single filtered subset) into a DataFrame.

    Preserved for backward compatibility. For bounded ingestion prefer
    pull_query_batches_to_dataframe() with a strategy-built query list.
    """
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


def stream_collection_to_parquet(
    uri_env_var: str,
    db_env_var: str,
    collection_env_var: str,
    output_paths: list[Path],
    query_filter: dict | None = None,
) -> dict[str, Any]:
    """Stream a single MongoDB query to parquet file(s).

    Preserved for backward compatibility. For bounded ingestion prefer
    stream_query_batches_to_parquet() with a strategy-built query list.
    """
    return stream_query_batches_to_parquet(
        uri_env_var=uri_env_var,
        db_env_var=db_env_var,
        collection_env_var=collection_env_var,
        output_paths=output_paths,
        query_filters=[query_filter or {}],
    )
