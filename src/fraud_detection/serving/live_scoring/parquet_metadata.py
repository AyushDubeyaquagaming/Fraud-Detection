from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fraud_detection.components.feature_engineering import (
    TIMESTAMP_CANDIDATES,
    _normalize_timestamp,
)
from fraud_detection.logger import get_logger

logger = get_logger(__name__)

PARQUET_METADATA_SCAN_BATCH_SIZE = 100_000


def _to_utc_datetime(value: object) -> datetime | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def read_training_parquet_bounds(parquet_path: Path) -> tuple[datetime | None, datetime | None]:
    """Return (min_ts, max_ts) across all available timestamp candidate columns
    without loading row data.

    Mirrors the training-time fallback chain in `_normalize_timestamp` — bounds
    are unioned across every candidate timestamp column that exists in the
    parquet, so a row that only has `createdAt` (and a NaT `trans_date`) still
    contributes to the bounds.

    Strategy
    --------
    1) Read row-group statistics from the parquet footer (no data scan) for
       each candidate column and union them.
    2) If statistics are missing for any candidate, fall back to a streamed
       single-column dataset scan over all candidates — never holds more than
       one batch in memory.
    3) Return (None, None) if no timestamp column can be resolved.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return None, None

    try:
        pf = pq.ParquetFile(str(parquet_path))
    except Exception as exc:
        logger.warning("Could not open parquet at %s for metadata read: %s", parquet_path, exc)
        return None, None

    schema_names = pf.schema_arrow.names
    available = [c for c in TIMESTAMP_CANDIDATES if c in schema_names]
    if not available:
        return None, None

    overall_min: datetime | None = None
    overall_max: datetime | None = None
    metadata_complete = True
    for column_name in available:
        bounds = _read_bounds_from_row_group_statistics(pf, column_name)
        if bounds is None:
            metadata_complete = False
            break
        col_min, col_max = bounds
        if col_min is not None and (overall_min is None or col_min < overall_min):
            overall_min = col_min
        if col_max is not None and (overall_max is None or col_max > overall_max):
            overall_max = col_max

    if metadata_complete and overall_min is not None and overall_max is not None:
        return overall_min, overall_max

    return _read_bounds_from_streamed_scan(parquet_path, available)


def _read_bounds_from_row_group_statistics(
    pf: pq.ParquetFile,
    column_name: str,
) -> tuple[datetime | None, datetime | None] | None:
    try:
        col_idx = pf.schema_arrow.get_field_index(column_name)
        if col_idx < 0:
            return None
        overall_min = None
        overall_max = None
        for rg_idx in range(pf.num_row_groups):
            col_meta = pf.metadata.row_group(rg_idx).column(col_idx)
            stats = col_meta.statistics
            if stats is None or not stats.has_min_max:
                return None
            rg_min = stats.min
            rg_max = stats.max
            if overall_min is None or rg_min < overall_min:
                overall_min = rg_min
            if overall_max is None or rg_max > overall_max:
                overall_max = rg_max
    except Exception as exc:
        logger.debug("Row-group statistics unavailable on %s: %s", column_name, exc)
        return None

    ts_min = _to_utc_datetime(overall_min)
    ts_max = _to_utc_datetime(overall_max)
    if ts_min is None or ts_max is None:
        return None
    return ts_min, ts_max


def _read_bounds_from_streamed_scan(
    parquet_path: Path,
    candidate_columns: list[str],
) -> tuple[datetime | None, datetime | None]:
    if not candidate_columns:
        return None, None

    overall_min: pd.Timestamp | None = None
    overall_max: pd.Timestamp | None = None

    try:
        dataset = ds.dataset(parquet_path, format="parquet")
        scanner = dataset.scanner(columns=candidate_columns, batch_size=PARQUET_METADATA_SCAN_BATCH_SIZE)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            batch_df = batch.to_pandas()
            ts = _normalize_timestamp(batch_df).dropna()
            if ts.empty:
                continue
            b_min = ts.min()
            b_max = ts.max()
            if overall_min is None or b_min < overall_min:
                overall_min = b_min
            if overall_max is None or b_max > overall_max:
                overall_max = b_max
    except Exception as exc:
        logger.warning("Streamed bounds scan failed for %s: %s", parquet_path, exc)
        return None, None

    if overall_min is None or overall_max is None:
        return None, None
    return overall_min.to_pydatetime(), overall_max.to_pydatetime()
