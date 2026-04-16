from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)


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

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("DataIngestion: starting (source=%s)", self.config.source)
        try:
            ensure_dir(self.config.output_dir)
            raw_path = self.config.output_dir / "raw_data.parquet"

            if self.config.source == "parquet":
                df = self._ingest_from_parquet(raw_path)
                ingestion_stats = self._summarize_dataframe(df)
            elif self.config.source == "mongodb":
                ingestion_stats = self._ingest_from_mongodb(raw_path)
            else:
                raise ValueError(f"Unknown source: {self.config.source}")

            row_count = ingestion_stats["row_count"]
            member_count = ingestion_stats["member_count"]
            date_range = ingestion_stats["date_range"]

            report = {
                "source_type": self.config.source,
                "row_count": row_count,
                "member_count": member_count,
                "date_range": date_range,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            report_path = self.config.output_dir / "ingestion_report.json"
            write_json(report, report_path)

            logger.info(
                "DataIngestion: complete — %d rows, %d members, saved to %s",
                row_count, member_count, raw_path,
            )
            return DataIngestionArtifact(
                raw_data_path=raw_path,
                ingestion_report_path=report_path,
                row_count=row_count,
                member_count=member_count,
                source_type=self.config.source,
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def _ingest_from_parquet(self, output_path: Path) -> pd.DataFrame:
        src = Path(self.config.parquet_path)
        if not src.exists():
            raise FileNotFoundError(f"Parquet source not found: {src}")
        logger.info("Copying parquet from %s → %s", src, output_path)
        shutil.copy2(src, output_path)
        df = pd.read_parquet(output_path)
        return df

    def _ingest_from_mongodb(self, output_path: Path) -> dict:
        from fraud_detection.utils.mongodb import stream_collection_to_parquet

        cache_path = Path(self.config.parquet_path)
        output_paths = [output_path]
        if cache_path != output_path:
            output_paths.append(cache_path)

        stats = stream_collection_to_parquet(
            uri_env_var=self.config.mongo_uri_env_var,
            db_env_var=self.config.mongo_database_env_var,
            collection_env_var=self.config.mongo_collection_env_var,
            output_paths=output_paths,
        )
        if cache_path != output_path:
            logger.info("Refreshed cached parquet from live MongoDB at %s", cache_path)
        return stats
