from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fraud_detection.constants.constants import BATCH_SCORING_CONFIG_FILE_PATH, MODEL_BUNDLE_FILE, REPO_ROOT
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
from fraud_detection.utils.common import load_joblib, read_json, read_yaml, write_json
from fraud_detection.utils.mongodb import MONGO_PROJECTION, get_serving_mongo_collection
from fraud_detection.utils.per_draw_recall import build_batch_false_negative_report

logger = get_logger(__name__)

PARQUET_SUMMARY_BATCH_SIZE = 250_000
BATCH_SCORING_STREAM_BATCH_SIZE = 100_000
BATCH_SCORING_OUTPUT_BATCH_SIZE = 1_000
MONGO_BATCH_SCORING_CHUNK_DAYS = 1


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_window_bounds(window: dict[str, object], timestamp_min: pd.Timestamp, timestamp_max: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = window.get("start_date")
    end_date = window.get("end_date")
    lookback_days = window.get("lookback_days")

    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else timestamp_max
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
    elif lookback_days is not None:
        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
    else:
        start_ts = timestamp_min
    if start_ts > end_ts:
        raise ValueError(f"Invalid batch scoring window: start {start_ts} is after end {end_ts}")
    return start_ts, end_ts


def _resolve_live_window_bounds(
    window: dict[str, object],
    *,
    now: datetime | None = None,
) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    timestamp_field = str(window.get("timestamp_field", "trans_date"))
    anchor = pd.Timestamp(now or datetime.now(timezone.utc))
    anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
    start_date = window.get("start_date")
    end_date = window.get("end_date")
    lookback_days = window.get("lookback_days")

    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else anchor
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
    elif lookback_days is not None:
        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
    else:
        raise ValueError("Mongo-backed batch scoring requires start_date/end_date or lookback_days in batch_scoring.window")
    if start_ts > end_ts:
        raise ValueError(f"Invalid batch scoring window: start {start_ts} is after end {end_ts}")
    return timestamp_field, start_ts, end_ts


def _diagnostic_window_bounds(
    *,
    batch_source: str,
    window: dict[str, object],
    source_cfg: dict[str, object],
    now: datetime,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if batch_source == "mongodb":
        _, start_ts, end_ts = _resolve_live_window_bounds(window, now=now)
        return start_ts, end_ts
    if batch_source == "candidate_store":
        if not window.get("start_date") or not window.get("end_date"):
            return None
        return pd.Timestamp(window["start_date"], tz="UTC"), pd.Timestamp(window["end_date"], tz="UTC")
    if source_cfg.get("source") == "parquet":
        timestamp_field = str(window.get("timestamp_field", "trans_date"))
        min_ts, max_ts = _timestamp_bounds(_resolve_repo_path(str(source_cfg["parquet_path"])), timestamp_field)
        if max_ts is None:
            return None
        return _resolve_window_bounds(window, min_ts or max_ts, max_ts)
    return None


def _to_filter_bound(value: pd.Timestamp, arrow_type: pa.DataType) -> object:
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz:
            return value.tz_convert(arrow_type.tz).to_pydatetime()
        return value.tz_convert("UTC").tz_localize(None).to_pydatetime()
    if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
        return value.tz_convert("UTC").tz_localize(None).date()
    raise TypeError(f"Unsupported parquet timestamp type for window filtering: {arrow_type}")


def _timestamp_bounds(parquet_path: Path, timestamp_field: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema_arrow.names
    if timestamp_field not in schema_names:
        raise ValueError(f"Timestamp field '{timestamp_field}' is missing from parquet source {parquet_path}")

    min_ts = None
    max_ts = None
    for batch in parquet_file.iter_batches(columns=[timestamp_field], batch_size=PARQUET_SUMMARY_BATCH_SIZE):
        parsed = pd.to_datetime(batch.column(0).to_pandas(), errors="coerce", utc=True)
        valid = parsed.dropna()
        if valid.empty:
            continue
        batch_min = valid.min()
        batch_max = valid.max()
        min_ts = batch_min if min_ts is None or batch_min < min_ts else min_ts
        max_ts = batch_max if max_ts is None or batch_max > max_ts else max_ts
    return min_ts, max_ts


def _iter_candidate_store_batches(store_path: Path, *, window: dict[str, object]):
    if not store_path.exists():
        raise FileNotFoundError(f"Candidate store not found: {store_path}")
    dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
    filter_expr = None
    if window.get("start_date"):
        filter_expr = ds.field("trans_date_min") >= pd.Timestamp(window["start_date"], tz="UTC").to_pydatetime()
    if window.get("end_date"):
        end_expr = ds.field("trans_date_min") < pd.Timestamp(window["end_date"], tz="UTC").to_pydatetime()
        filter_expr = end_expr if filter_expr is None else filter_expr & end_expr
    scanner = dataset.scanner(filter=filter_expr, batch_size=BATCH_SCORING_STREAM_BATCH_SIZE)
    row_count = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        row_count += len(frame)
        yield frame
    if row_count == 0:
        raise ValueError(f"Candidate store window returned 0 rows from {store_path}")


def _iter_mongo_draw_groups(
    *,
    mongo_config: dict[str, object],
    window: dict[str, object],
    now: datetime | None = None,
):
    timestamp_field, start_ts, end_ts = _resolve_live_window_bounds(window, now=now)
    chunk_days = max(1, int(window.get("chunk_days", MONGO_BATCH_SCORING_CHUNK_DAYS)))
    logger.info(
        "Batch scoring mongo window: %s in [%s, %s) chunk_days=%s",
        timestamp_field,
        start_ts.isoformat(),
        end_ts.isoformat(),
        chunk_days,
    )
    collection = get_serving_mongo_collection(
        str(mongo_config.get("uri_env_var", "MONGODB_URI")),
        str(mongo_config.get("database_env_var", "MONGODB_DATABASE")),
        str(mongo_config.get("collection_env_var", "MONGODB_COLLECTION_ROULETTE_REPORT")),
    )
    current_draw_id = None
    pending_rows: list[dict] = []
    row_count = 0

    slice_start = start_ts
    while slice_start < end_ts:
        slice_end = min(slice_start + pd.Timedelta(days=chunk_days), end_ts)
        query = {
            timestamp_field: {
                "$gte": slice_start.to_pydatetime(),
                "$lt": slice_end.to_pydatetime(),
            }
        }
        logger.info(
            "Batch scoring mongo chunk: %s in [%s, %s)",
            timestamp_field,
            slice_start.isoformat(),
            slice_end.isoformat(),
        )
        cursor = (
            collection.find(query, MONGO_PROJECTION)
            .sort([(timestamp_field, 1), ("draw_id", 1)])
            .batch_size(BATCH_SCORING_STREAM_BATCH_SIZE)
        )
        for row in cursor:
            draw_id = pd.to_numeric(row.get("draw_id"), errors="coerce")
            if pd.isna(draw_id):
                continue
            draw_id = int(draw_id)
            if current_draw_id is None:
                current_draw_id = draw_id
            if draw_id != current_draw_id:
                yield pd.DataFrame(pending_rows)
                pending_rows = []
                current_draw_id = draw_id
            pending_rows.append(row)
            row_count += 1
        slice_start = slice_end
    if pending_rows:
        yield pd.DataFrame(pending_rows)
    if row_count == 0:
        raise ValueError("Mongo-backed batch scoring window returned 0 rows")


def _iter_parquet_draw_groups(parquet_path: Path, *, window: dict[str, object], timestamp_field: str):
    dataset = ds.dataset(parquet_path, format="parquet")
    if timestamp_field not in dataset.schema.names:
        raise ValueError(f"Timestamp field '{timestamp_field}' is missing from parquet source {parquet_path}")

    min_ts, max_ts = _timestamp_bounds(parquet_path, timestamp_field)
    if max_ts is None:
        raise ValueError(f"No parseable timestamps found in parquet column '{timestamp_field}'")

    start_ts, end_ts = _resolve_window_bounds(window, min_ts or max_ts, max_ts)
    arrow_type = dataset.schema.field(timestamp_field).type
    filter_expr = (
        (ds.field(timestamp_field) >= _to_filter_bound(start_ts, arrow_type))
        & (ds.field(timestamp_field) < _to_filter_bound(end_ts, arrow_type))
    )
    logger.info(
        "Batch scoring parquet window: %s in [%s, %s) from %s",
        timestamp_field,
        start_ts.isoformat(),
        end_ts.isoformat(),
        parquet_path,
    )
    pending = pd.DataFrame()
    saw_rows = False
    scanner = dataset.scanner(filter=filter_expr, batch_size=BATCH_SCORING_STREAM_BATCH_SIZE)
    for batch in scanner.to_batches():
        chunk = batch.to_pandas()
        if chunk.empty:
            continue
        saw_rows = True
        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        draw_ids = pd.to_numeric(chunk["draw_id"], errors="coerce")
        chunk = chunk.loc[draw_ids.notna()].copy()
        if chunk.empty:
            pending = pd.DataFrame(columns=batch.schema.names)
            continue
        chunk["draw_id"] = draw_ids.loc[chunk.index].astype(int)
        last_draw_id = int(chunk["draw_id"].iloc[-1])
        ready = chunk.loc[chunk["draw_id"].ne(last_draw_id)].copy()
        for _, draw_rows in ready.groupby("draw_id", sort=False):
            yield draw_rows
        pending = chunk.loc[chunk["draw_id"].eq(last_draw_id)].copy()
    if not pending.empty:
        yield pending
        saw_rows = True
    if not saw_rows:
        raise ValueError(f"Batch scoring window returned 0 rows from {parquet_path}")


class _ParquetDocWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def write(self, docs: list[dict]) -> None:
        if not docs:
            return
        table = pa.Table.from_pylist(docs)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(str(self.path), self.schema)
        else:
            table = table.cast(self.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class BatchScoringPipeline:
    """Replay draw-level partnership scoring over a parquet window."""

    def __init__(self, config_path: Path = BATCH_SCORING_CONFIG_FILE_PATH):
        self.config_path = config_path

    def run(self) -> Path:
        logger.info("PartnershipBatchScoringPipeline: starting")
        try:
            config = read_yaml(self.config_path)
            current_dir = _resolve_repo_path(config["pipeline"]["current_dir"])
            bundle_path = current_dir / MODEL_BUNDLE_FILE
            if not bundle_path.exists():
                raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
            bundle = load_joblib(bundle_path)
            batch_cfg = config.get("batch_scoring", {}) or {}
            partnership_cfg = config.get("partnership", {})
            use_candidate_store = bool(partnership_cfg.get("use_candidate_store", bundle.get("use_candidate_store", False)))
            source_cfg = config["data_ingestion"]
            batch_source = str(batch_cfg.get("source", "mongodb" if use_candidate_store else source_cfg.get("source", "parquet"))).lower()
            window = batch_cfg.get("window", {}) or {}
            if batch_source == "candidate_store":
                iterator = _iter_candidate_store_batches(
                    _resolve_repo_path(partnership_cfg.get("candidate_store_path", "data_store/candidate_draws")),
                    window=window,
                )
                score_mode = "candidate_store"
            elif batch_source == "mongodb":
                iterator = _iter_mongo_draw_groups(
                    mongo_config=source_cfg.get("mongodb", {}) or {},
                    window=window,
                )
                score_mode = "raw_rows"
            else:
                if source_cfg["source"] != "parquet":
                    raise ValueError("Partnership batch scoring currently expects a parquet source.")
                timestamp_field = str(window.get("timestamp_field", "trans_date"))
                iterator = _iter_parquet_draw_groups(
                    _resolve_repo_path(source_cfg["parquet_path"]),
                    window=window,
                    timestamp_field=timestamp_field,
                )
                score_mode = "raw_rows"

            manifest_path = current_dir / str(config.get("serving", {}).get("manifest_file", "serving_manifest.json"))
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            partnership_table_path = current_dir / str(manifest.get("partnership_table_file", "partnership_table.parquet"))
            partnership_table = pd.read_parquet(partnership_table_path) if partnership_table_path.exists() else pd.DataFrame()
            ccs_table_path = current_dir / str(manifest.get("ccs_concentration_table_file", "ccs_concentration_table.parquet"))
            ccs_table = pd.read_parquet(ccs_table_path) if ccs_table_path.exists() else pd.DataFrame()
            scorer = DrawScorer(
                bundle,
                source_run_id=manifest.get("run_id") or bundle.get("source_run_id"),
                partnership_table=partnership_table,
                ccs_concentration_table=ccs_table,
            )
            output_path = current_dir / "live_predictions_backfill.parquet"
            writer = _ParquetDocWriter(output_path)
            pending_docs: list[dict] = []
            draw_count = 0
            try:
                for item in iterator:
                    try:
                        results = scorer.score_candidate_batch(item) if score_mode == "candidate_store" else [scorer.score_draw(item)]
                        pending_docs.extend(result.to_mongo_doc() for result in results)
                        draw_count += len(results)
                        if len(pending_docs) >= BATCH_SCORING_OUTPUT_BATCH_SIZE:
                            writer.write(pending_docs)
                            pending_docs.clear()
                    except Exception as exc:
                        draw_id = (
                            item.get("draw_id", pd.Series(dtype=object)).iloc[0]
                            if score_mode == "candidate_store"
                            else item.get("draw_id", pd.NA).iloc[0]
                        )
                        logger.warning("Skipping draw_id=%s during batch scoring: %s", draw_id, exc)
                writer.write(pending_docs)
            finally:
                writer.close()
            logger.info("Batch scoring: scored %s draws", draw_count)
            if draw_count == 0 and not output_path.exists():
                pd.DataFrame().to_parquet(output_path, index=False)
            report = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "draws_scored": int(draw_count),
                "output_path": str(output_path),
                "model_version": bundle.get("model_version", "partnership_v1"),
                "source": batch_source,
                "window": window,
                "filter_parity": (
                    "raw_rows are converted through DrawScorer._raw_draw_to_candidate_row before pair scanning"
                    if score_mode == "raw_rows"
                    else "candidate_store rows are scored directly through the pair scanner"
                ),
            }
            try:
                fraud_csv_path = _resolve_repo_path(
                    str((config.get("data_validation", {}) or {}).get("fraud_csv_path", "ROULET CHEATING DATA.csv"))
                )
                diagnostic_bounds = _diagnostic_window_bounds(
                    batch_source=batch_source,
                    window=window,
                    source_cfg=source_cfg,
                    now=datetime.now(timezone.utc),
                )
                if fraud_csv_path.exists() and diagnostic_bounds is not None:
                    false_negative_report = build_batch_false_negative_report(
                        fraud_csv_path=fraud_csv_path,
                        scored_predictions_path=output_path,
                        window_start=diagnostic_bounds[0],
                        window_end=diagnostic_bounds[1],
                        output_path=current_dir / "batch_false_negative_report.json",
                    )
                    report["false_negative_report_path"] = str(current_dir / "batch_false_negative_report.json")
                    report["false_negative_draws"] = false_negative_report.get("false_negative_draws", 0)
                else:
                    report["false_negative_report_status"] = "skipped"
            except Exception as diag_exc:
                logger.warning("Batch false-negative diagnostics failed: %s", diag_exc)
                report["false_negative_report_status"] = f"failed: {diag_exc}"
            write_json(report, current_dir / "batch_scoring_report.json")
            return current_dir
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc
