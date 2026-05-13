from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fraud_detection.constants.constants import (
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.mongodb import (
    MONGO_PROJECTION,
    TIMESTAMP_COLUMNS,
    build_query_batches_from_strategy,
    get_mongo_collection,
    _normalize_batch_for_parquet,
)

logger = get_logger(__name__)

ROLLING_STORE_BATCH_SIZE = 10_000
SCHEMA_FILE = "_schema.json"


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _partition_name(day: pd.Timestamp) -> str:
    return f"trans_date={day.strftime('%Y-%m-%d')}"


def _write_schema(schema: pa.Schema, output_root: Path) -> None:
    payload = {
        "fields": [{"name": field.name, "type": str(field.type)} for field in schema],
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_root / SCHEMA_FILE).open("w") as f:
        json.dump(payload, f, indent=2)


def _read_schema(output_root: Path) -> pa.Schema | None:
    if not output_root.exists():
        return None
    part_files = sorted(output_root.glob("trans_date=*/*.parquet"))
    if not part_files:
        return None
    return pq.ParquetFile(part_files[0]).schema_arrow


def _normalize_for_store(df: pd.DataFrame) -> pd.DataFrame:
    projected_columns = list(MONGO_PROJECTION.keys())
    working = df.reindex(columns=projected_columns)
    return _normalize_batch_for_parquet(working.to_dict(orient="records"), projected_columns)


def _timestamp_stats() -> dict[str, dict[str, Any]]:
    return {column: {"min": None, "max": None} for column in TIMESTAMP_COLUMNS}


def _update_stats(
    df: pd.DataFrame,
    member_ids: set[Any],
    timestamp_stats: dict[str, dict[str, Any]],
) -> None:
    if "member_id" in df.columns:
        member_ids.update(df["member_id"].dropna().tolist())
    for column in TIMESTAMP_COLUMNS:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
        valid = parsed.dropna()
        if valid.empty:
            continue
        current = timestamp_stats[column]
        batch_min = valid.min()
        batch_max = valid.max()
        if current["min"] is None or batch_min < current["min"]:
            current["min"] = batch_min
        if current["max"] is None or batch_max > current["max"]:
            current["max"] = batch_max


def _date_range(timestamp_stats: dict[str, dict[str, Any]]) -> dict[str, str]:
    for column in TIMESTAMP_COLUMNS:
        current = timestamp_stats[column]
        if current["min"] is not None and current["max"] is not None:
            return {"from": str(current["min"]), "to": str(current["max"])}
    return {}


def _write_normalized_partitioned_dataframe(
    df: pd.DataFrame,
    output_root: Path,
    schema: pa.Schema | None = None,
) -> pa.Schema:
    if df.empty:
        raise ValueError("Cannot write an empty rolling-store DataFrame.")
    if "trans_date" not in df.columns:
        raise ValueError("rolling parquet store requires a trans_date column.")

    working = df.copy()
    working["_partition_day"] = pd.to_datetime(working["trans_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    working = working.loc[working["_partition_day"].notna()].copy()
    if working.empty:
        raise ValueError("No rows had a valid trans_date for partitioning.")

    output_root.mkdir(parents=True, exist_ok=True)
    active_schema = schema
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for day, day_df in working.groupby("_partition_day", sort=True):
        part_dir = output_root / f"trans_date={day}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(day_df.drop(columns="_partition_day"), preserve_index=False)
        if active_schema is None:
            active_schema = table.schema
        else:
            table = table.cast(active_schema)
        pq.write_table(table, part_dir / f"part-{timestamp}.parquet")

    if active_schema is None:
        raise ValueError("No schema was produced while writing rolling store.")
    _write_schema(active_schema, output_root)
    return active_schema


def _write_partitioned_dataframe(df: pd.DataFrame, output_root: Path, schema: pa.Schema | None = None) -> pa.Schema:
    return _write_normalized_partitioned_dataframe(_normalize_for_store(df), output_root, schema=schema)


def write_dataframe_to_rolling_store(df: pd.DataFrame, output_root: Path) -> None:
    """Test/helper entry point for writing an in-memory frame to the store."""
    output_root = Path(output_root)
    schema = _read_schema(output_root)
    _write_partitioned_dataframe(df, output_root, schema=schema)


def _stream_window_to_rolling_store(
    start_date: Any,
    end_date: Any,
    output_root: Path,
    schema: pa.Schema | None = None,
) -> dict[str, Any]:
    query_filters = build_query_batches_from_strategy(
        "date_window",
        {
            "timestamp_field": "trans_date",
            "start_date": _utc_timestamp(start_date).isoformat(),
            "end_date": _utc_timestamp(end_date).isoformat(),
        },
    )
    projected_columns = list(MONGO_PROJECTION.keys())
    client, collection = get_mongo_collection(ENV_MONGODB_URI, ENV_MONGODB_DATABASE, ENV_MONGODB_COLLECTION)
    active_schema = schema
    row_count = 0
    member_ids: set[Any] = set()
    stats = _timestamp_stats()

    def flush_batch(batch_docs: list[dict[str, Any]]) -> None:
        nonlocal active_schema, row_count
        if not batch_docs:
            return
        batch_df = _normalize_batch_for_parquet(batch_docs, projected_columns)
        if batch_df.empty:
            return
        active_schema = _write_normalized_partitioned_dataframe(batch_df, output_root, schema=active_schema)
        row_count += len(batch_df)
        _update_stats(batch_df, member_ids, stats)

    try:
        for query_filter in query_filters:
            cursor = collection.find(query_filter, MONGO_PROJECTION, no_cursor_timeout=True).batch_size(
                ROLLING_STORE_BATCH_SIZE
            )
            try:
                batch_docs: list[dict[str, Any]] = []
                for doc in cursor:
                    batch_docs.append(doc)
                    if len(batch_docs) >= ROLLING_STORE_BATCH_SIZE:
                        flush_batch(batch_docs)
                        batch_docs = []
                flush_batch(batch_docs)
            finally:
                cursor.close()
    finally:
        client.close()

    if row_count == 0:
        raise ValueError("Rolling-store Mongo query returned 0 documents.")
    return {
        "row_count": row_count,
        "member_count": len(member_ids),
        "date_range": _date_range(stats),
        "partition_count": len(list(output_root.glob("trans_date=*"))),
        "schema_fields": active_schema.names if active_schema is not None else [],
    }


def bootstrap_rolling_window(start_date: Any, end_date: Any, output_root: Path) -> dict[str, Any]:
    """One-shot bootstrap of a partitioned trans_date rolling store."""
    try:
        output_root = Path(output_root)
        tmp_root = output_root.parent / f".tmp_bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        stats = _stream_window_to_rolling_store(start_date, end_date, tmp_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        tmp_root.replace(output_root)
        return {
            "status": "bootstrapped",
            "output_root": str(output_root),
            "row_count": int(stats["row_count"]),
            "member_count": int(stats["member_count"]),
            "date_range": stats["date_range"],
            "partition_count": len(list(output_root.glob("trans_date=*"))),
            "schema_fields": stats["schema_fields"],
        }
    except Exception as exc:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise FraudDetectionException(exc, sys) from exc


def update_rolling_window(output_root: Path, lookback_days: int = 90, days_to_pull: int = 7) -> dict[str, Any]:
    """Append recent Mongo rows and remove partitions older than lookback_days."""
    try:
        output_root = Path(output_root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tmp_root = output_root.parent / f".tmp_update_{timestamp}"
        backup_root = output_root.parent / f".backup_update_{timestamp}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        if output_root.exists():
            shutil.copytree(output_root, tmp_root)
        else:
            tmp_root.mkdir(parents=True, exist_ok=True)

        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - pd.Timedelta(days=int(days_to_pull))
        stats = _stream_window_to_rolling_store(start_date, end_date, tmp_root, schema=_read_schema(tmp_root))

        cutoff = (end_date - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
        removed = []
        for part_dir in tmp_root.glob("trans_date=*"):
            day = part_dir.name.split("=", 1)[1]
            if day < cutoff:
                shutil.rmtree(part_dir)
                removed.append(day)

        swapped = False
        try:
            if output_root.exists():
                output_root.rename(backup_root)
            tmp_root.rename(output_root)
            swapped = True
        finally:
            if swapped:
                shutil.rmtree(backup_root, ignore_errors=True)
            else:
                if backup_root.exists() and not output_root.exists():
                    backup_root.rename(output_root)
                shutil.rmtree(tmp_root, ignore_errors=True)
        return {
            "status": "updated",
            "output_root": str(output_root),
            "rows_pulled": int(stats["row_count"]),
            "member_count": int(stats["member_count"]),
            "date_range": stats["date_range"],
            "partition_count": len(list(output_root.glob("trans_date=*"))),
            "removed_partitions": removed,
        }
    except Exception as exc:
        for candidate in locals().get("tmp_root", None), locals().get("backup_root", None):
            if candidate is not None:
                shutil.rmtree(candidate, ignore_errors=True)
        raise FraudDetectionException(exc, sys) from exc


def read_rolling_window(output_root: Path, start_date: Any | None = None, end_date: Any | None = None) -> ds.Dataset:
    """Return a pyarrow dataset for the partitioned rolling store."""
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Rolling parquet store not found: {output_root}")
    # Do not enable hive partitioning here: the partition directory is named
    # trans_date=YYYY-MM-DD and the parquet files also contain a timestamp
    # column named trans_date. Let the file schema win and filter after read.
    return ds.dataset(output_root, format="parquet", partitioning=None)


def materialize_rolling_window_to_parquet(
    output_root: Path,
    output_path: Path,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> dict[str, Any]:
    """Read the rolling store and write one consolidated parquet for pipeline compatibility."""
    dataset = read_rolling_window(output_root, start_date=start_date, end_date=end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    row_count = 0
    member_ids: set[Any] = set()
    stats = _timestamp_stats()
    try:
        for batch in dataset.to_batches(batch_size=ROLLING_STORE_BATCH_SIZE):
            df = batch.to_pandas()
            if df.empty:
                continue
            if start_date is not None:
                df = df.loc[pd.to_datetime(df["trans_date"], errors="coerce", utc=True) >= _utc_timestamp(start_date)]
            if end_date is not None:
                df = df.loc[pd.to_datetime(df["trans_date"], errors="coerce", utc=True) < _utc_timestamp(end_date)]
            if df.empty:
                continue
            _update_stats(df, member_ids, stats)
            row_count += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty = pd.DataFrame(columns=list(MONGO_PROJECTION.keys()))
        empty.to_parquet(output_path, index=False)

    return {
        "row_count": int(row_count),
        "member_count": len(member_ids),
        "date_range": _date_range(stats) if row_count else {"from": None, "to": None},
        "query_count": 0,
    }
