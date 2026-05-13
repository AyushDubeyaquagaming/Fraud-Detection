from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)

PARQUET_SUMMARY_BATCH_SIZE = 100_000
PARQUET_WINDOW_BATCH_SIZE = 100_000


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _summarize_dataframe(df: pd.DataFrame) -> dict:
        row_count = len(df)
        member_count = int(df["member_id"].nunique()) if "member_id" in df.columns else 0

        date_range: dict = {}
        for col in ["createdAt", "trans_date", "updatedAt"]:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                valid = parsed.dropna()
                if not valid.empty:
                    date_range = {
                        "from": str(valid.min()),
                        "to": str(valid.max()),
                    }
                    break

        return {
            "row_count": row_count,
            "member_count": member_count,
            "date_range": date_range,
        }

    @staticmethod
    def _summarize_parquet(parquet_path: Path) -> dict:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = parquet_file.schema_arrow.names
        row_count = parquet_file.metadata.num_rows

        member_ids: set[str] = set()
        if "member_id" in schema_names:
            for batch in parquet_file.iter_batches(
                columns=["member_id"],
                batch_size=PARQUET_SUMMARY_BATCH_SIZE,
            ):
                member_ids.update(
                    str(value)
                    for value in batch.column(0).to_pylist()
                    if value is not None
                )

        date_range: dict = {}
        for col in ["createdAt", "trans_date", "updatedAt"]:
            if col not in schema_names:
                continue

            min_ts = None
            max_ts = None
            for batch in parquet_file.iter_batches(
                columns=[col],
                batch_size=PARQUET_SUMMARY_BATCH_SIZE,
            ):
                parsed = pd.to_datetime(batch.column(0).to_pandas(), errors="coerce", utc=True)
                valid = parsed.dropna()
                if valid.empty:
                    continue
                batch_min = valid.min()
                batch_max = valid.max()
                min_ts = batch_min if min_ts is None or batch_min < min_ts else min_ts
                max_ts = batch_max if max_ts is None or batch_max > max_ts else max_ts

            if min_ts is not None and max_ts is not None:
                date_range = {
                    "from": str(min_ts),
                    "to": str(max_ts),
                }
                break

        return {
            "row_count": row_count,
            "member_count": len(member_ids),
            "date_range": date_range,
        }

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("DataIngestion: starting (source=%s)", self.config.source)
        try:
            ensure_dir(self.config.output_dir)
            raw_path = self.config.output_dir / "raw_data.parquet"

            if self.config.source == "parquet":
                ingestion_stats = self._ingest_from_parquet(raw_path)
                strategy_used = self.config.parquet_strategy if self.config.parquet_strategy != "full_copy" else None
                query_count = 1
            elif self.config.source == "mongodb":
                ingestion_stats = self._ingest_from_mongodb(raw_path)
                strategy_used = self.config.mongo_strategy
                query_count = ingestion_stats.get("query_count", 1)
            else:
                raise ValueError(f"Unknown source: {self.config.source}")

            row_count = ingestion_stats["row_count"]
            member_count = ingestion_stats["member_count"]
            date_range = ingestion_stats.get("date_range", {})

            report = {
                "source_type": self.config.source,
                "row_count": row_count,
                "member_count": member_count,
                "date_range": date_range,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            if strategy_used is not None:
                report["strategy_used"] = strategy_used
                report["query_count"] = query_count

            report_path = self.config.output_dir / "ingestion_report.json"
            write_json(report, report_path)

            logger.info(
                "DataIngestion: complete — %d rows, %d members, strategy=%s, query_count=%d, saved to %s",
                row_count,
                member_count,
                strategy_used or "n/a",
                query_count,
                raw_path,
            )
            return DataIngestionArtifact(
                raw_data_path=raw_path,
                ingestion_report_path=report_path,
                row_count=row_count,
                member_count=member_count,
                source_type=self.config.source,
                strategy_used=strategy_used,
                query_count=query_count,
                date_range=date_range or None,
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def _ingest_from_parquet(self, output_path: Path) -> dict:
        src = Path(self.config.parquet_path)
        if not src.exists():
            raise FileNotFoundError(f"Parquet source not found: {src}")
        if self.config.parquet_strategy == "date_window":
            logger.info("Staging bounded parquet window from %s → %s", src, output_path)
            return self._stage_parquet_window(src, output_path, self.config.parquet_strategy_params)
        if self.config.parquet_strategy not in {"", "full_copy", None}:
            raise ValueError(f"Unknown parquet strategy: {self.config.parquet_strategy}")
        logger.info("Copying parquet from %s → %s", src, output_path)
        shutil.copy2(src, output_path)
        return self._summarize_parquet(output_path)

    @staticmethod
    def _to_filter_bound(value: pd.Timestamp, arrow_type: pa.DataType) -> object:
        if pa.types.is_timestamp(arrow_type):
            if arrow_type.tz:
                return value.tz_convert(arrow_type.tz).to_pydatetime()
            return value.tz_convert("UTC").tz_localize(None).to_pydatetime()
        if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
            return value.tz_convert("UTC").tz_localize(None).date()
        raise TypeError(f"Unsupported parquet timestamp type for window filtering: {arrow_type}")

    @staticmethod
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

    def _resolve_parquet_window(
        self,
        parquet_path: Path,
        strategy_params: dict,
    ) -> tuple[str, pd.Timestamp, pd.Timestamp]:
        timestamp_field = str(strategy_params.get("timestamp_field", "trans_date"))
        start_date = strategy_params.get("start_date")
        end_date = strategy_params.get("end_date")
        lookback_days = strategy_params.get("lookback_days")

        min_ts, max_ts = self._timestamp_bounds(parquet_path, timestamp_field)
        if max_ts is None:
            raise ValueError(f"No parseable timestamps found in parquet column '{timestamp_field}'")

        if end_date is not None:
            end_ts = pd.Timestamp(end_date, tz="UTC")
        else:
            end_ts = max_ts

        if start_date is not None:
            start_ts = pd.Timestamp(start_date, tz="UTC")
        elif lookback_days is not None:
            start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
        elif min_ts is not None:
            start_ts = min_ts
        else:
            start_ts = end_ts

        if start_ts > end_ts:
            raise ValueError(f"Invalid parquet date window: start {start_ts} is after end {end_ts}")
        return timestamp_field, start_ts, end_ts

    def _stage_parquet_window(self, parquet_path: Path, output_path: Path, strategy_params: dict) -> dict:
        timestamp_field, start_ts, end_ts = self._resolve_parquet_window(parquet_path, strategy_params)
        dataset = ds.dataset(parquet_path, format="parquet")
        if timestamp_field not in dataset.schema.names:
            raise ValueError(f"Timestamp field '{timestamp_field}' is missing from parquet schema")

        arrow_type = dataset.schema.field(timestamp_field).type
        filter_expr = (
            (ds.field(timestamp_field) >= self._to_filter_bound(start_ts, arrow_type))
            & (ds.field(timestamp_field) <= self._to_filter_bound(end_ts, arrow_type))
        )

        writer = None
        row_count = 0
        member_ids: set[str] = set()
        min_seen = None
        max_seen = None
        try:
            scanner = dataset.scanner(filter=filter_expr, batch_size=PARQUET_WINDOW_BATCH_SIZE)
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                table = pa.Table.from_batches([batch])
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(str(output_path), table.schema)
                writer.write_table(table)
                row_count += batch.num_rows

                if "member_id" in table.column_names:
                    member_ids.update(str(value) for value in table["member_id"].to_pylist() if value is not None)
                if timestamp_field in table.column_names:
                    parsed = pd.to_datetime(table[timestamp_field].to_pandas(), errors="coerce", utc=True).dropna()
                    if not parsed.empty:
                        batch_min = parsed.min()
                        batch_max = parsed.max()
                        min_seen = batch_min if min_seen is None or batch_min < min_seen else min_seen
                        max_seen = batch_max if max_seen is None or batch_max > max_seen else max_seen
        finally:
            if writer is not None:
                writer.close()

        if row_count == 0:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"Configured parquet window returned 0 rows for {timestamp_field} in [{start_ts.isoformat()}, {end_ts.isoformat()}]"
            )

        return {
            "row_count": row_count,
            "member_count": len(member_ids),
            "date_range": {
                "from": str(min_seen) if min_seen is not None else str(start_ts),
                "to": str(max_seen) if max_seen is not None else str(end_ts),
            },
        }

    def _ingest_from_mongodb(self, output_path: Path) -> dict:
        from fraud_detection.utils.mongodb import (
            build_query_batches_from_strategy,
            stream_query_batches_to_parquet,
        )
        from fraud_detection.utils.rolling_parquet_store import materialize_rolling_window_to_parquet

        strategy = self.config.mongo_strategy
        strategy_params = self.config.mongo_strategy_params

        logger.info(
            "DataIngestion (mongodb): strategy=%s, params=%s",
            strategy,
            strategy_params,
        )

        if strategy == "rolling_store":
            store_root = Path(strategy_params.get("store_root", "data_store/training_window"))
            lookback_days = int(strategy_params.get("lookback_days", 90))
            end_date = pd.Timestamp.now(tz="UTC")
            start_date = end_date - pd.Timedelta(days=lookback_days)
            stats = materialize_rolling_window_to_parquet(
                output_root=store_root,
                output_path=output_path,
                start_date=start_date,
                end_date=end_date,
            )
            return stats

        query_filters = build_query_batches_from_strategy(strategy, strategy_params)

        cache_path = Path(self.config.parquet_path)
        output_paths = [output_path]
        if cache_path != output_path:
            output_paths.append(cache_path)

        stats = stream_query_batches_to_parquet(
            uri_env_var=self.config.mongo_uri_env_var,
            db_env_var=self.config.mongo_database_env_var,
            collection_env_var=self.config.mongo_collection_env_var,
            output_paths=output_paths,
            query_filters=query_filters,
        )
        if cache_path != output_path:
            logger.info("Refreshed cached parquet from live MongoDB at %s", cache_path)
        return stats
