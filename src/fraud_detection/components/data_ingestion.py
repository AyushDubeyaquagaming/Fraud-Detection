from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)

PARQUET_SUMMARY_BATCH_SIZE = 100_000


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
                strategy_used = None
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
        logger.info("Copying parquet from %s → %s", src, output_path)
        shutil.copy2(src, output_path)
        return self._summarize_parquet(output_path)

    def _ingest_from_mongodb(self, output_path: Path) -> dict:
        from fraud_detection.utils.mongodb import (
            build_query_batches_from_strategy,
            stream_query_batches_to_parquet,
        )

        strategy = self.config.mongo_strategy
        strategy_params = self.config.mongo_strategy_params

        logger.info(
            "DataIngestion (mongodb): strategy=%s, params=%s",
            strategy,
            strategy_params,
        )

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
