from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)

VALIDATION_BATCH_SIZE = 100_000


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    @staticmethod
    def _count_nulls(parquet_file: pq.ParquetFile, column: str) -> int:
        null_count = 0
        for batch in parquet_file.iter_batches(columns=[column], batch_size=VALIDATION_BATCH_SIZE):
            null_count += batch.column(0).null_count
        return null_count

    @staticmethod
    def _sample_non_null_values(
        parquet_file: pq.ParquetFile,
        column: str,
        sample_size: int = 100,
    ) -> list:
        values: list = []
        for batch in parquet_file.iter_batches(columns=[column], batch_size=VALIDATION_BATCH_SIZE):
            for value in batch.column(0).to_pylist():
                if value is None:
                    continue
                values.append(value)
                if len(values) >= sample_size:
                    return values
        return values

    @staticmethod
    def _check_timestamp_parseable(
        parquet_file: pq.ParquetFile,
        schema_names: list[str],
        row_count: int,
    ) -> dict:
        timestamp_candidates = [
            "createdAt.$date", "trans_date.$date", "updatedAt.$date",
            "createdAt", "trans_date", "updatedAt", "ts",
        ]
        ts_col = next((column for column in timestamp_candidates if column in schema_names), None)
        if ts_col is None:
            return {"passed": True, "warning": "no timestamp column found — skipped"}

        field = parquet_file.schema_arrow.field(ts_col)
        if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
            null_count = DataValidation._count_nulls(parquet_file, ts_col)
            return {
                "passed": null_count < row_count,
                "column": ts_col,
                "null_count": null_count,
            }

        null_count = 0
        for batch in parquet_file.iter_batches(columns=[ts_col], batch_size=VALIDATION_BATCH_SIZE):
            parsed = pd.to_datetime(batch.column(0).to_pandas(), errors="coerce", utc=True)
            null_count += int(parsed.isna().sum())

        return {
            "passed": null_count < row_count,
            "column": ts_col,
            "null_count": null_count,
        }

    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("DataValidation: starting")
        try:
            ensure_dir(self.config.output_dir)
            parquet_file = pq.ParquetFile(self.ingestion_artifact.raw_data_path)
            row_count = parquet_file.metadata.num_rows
            schema_names = parquet_file.schema_arrow.names
            checks: dict[str, dict] = {}

            # 1. Row count
            checks["row_count"] = {
                "passed": row_count >= self.config.min_row_count,
                "actual": row_count,
                "minimum": self.config.min_row_count,
            }

            # 2. Required columns present
            missing_cols = [c for c in self.config.required_columns if c not in schema_names]
            checks["required_columns"] = {
                "passed": len(missing_cols) == 0,
                "missing": missing_cols,
            }

            # 3. member_id not null
            if "member_id" in schema_names:
                null_members = self._count_nulls(parquet_file, "member_id")
                checks["member_id_not_null"] = {"passed": null_members == 0, "null_count": null_members}
            else:
                checks["member_id_not_null"] = {"passed": False, "null_count": "column missing"}

            # 4. draw_id not null
            if "draw_id" in schema_names:
                null_draws = self._count_nulls(parquet_file, "draw_id")
                checks["draw_id_not_null"] = {"passed": null_draws == 0, "null_count": null_draws}
            else:
                checks["draw_id_not_null"] = {"passed": False, "null_count": "column missing"}

            # 5. bets column parseable (sample 100 rows)
            if "bets" in schema_names:
                sample = self._sample_non_null_values(parquet_file, "bets", sample_size=100)
                parse_errors = 0
                for val in sample:
                    if isinstance(val, str):
                        try:
                            json.loads(val)
                        except Exception:
                            parse_errors += 1
                checks["bets_parseable"] = {"passed": parse_errors == 0, "parse_errors_in_sample": parse_errors}
            else:
                checks["bets_parseable"] = {"passed": False, "error": "bets column missing"}

            # 6. Timestamp column parseable (handles plain names and .$date suffixed names)
            checks["timestamp_parseable"] = self._check_timestamp_parseable(
                parquet_file,
                schema_names,
                row_count,
            )

            # 7. Fraud CSV loadable
            fraud_csv_path = Path(self.config.fraud_csv_path)
            if fraud_csv_path.exists():
                try:
                    fraud_df = pd.read_csv(fraud_csv_path)
                    fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]
                    has_cols = all(c in fraud_df.columns for c in ["member_id", "draw_id"])
                    checks["fraud_csv"] = {"passed": has_cols, "rows": len(fraud_df)}
                except Exception as ex:
                    checks["fraud_csv"] = {"passed": False, "error": str(ex)}
            else:
                checks["fraud_csv"] = {"passed": False, "error": f"not found: {fraud_csv_path}"}

            failed = [k for k, v in checks.items() if not v.get("passed", False)]
            all_passed = len(failed) == 0

            report = {
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "all_passed": all_passed,
                "failed_checks": failed,
                "checks": checks,
            }
            report_path = self.config.output_dir / "validation_report.json"
            write_json(report, report_path)

            if not all_passed:
                msg = f"DataValidation failed checks: {failed}"
                logger.error(msg)
                raise FraudDetectionException(msg, sys)

            logger.info("DataValidation: all checks passed")
            return DataValidationArtifact(
                validation_report_path=report_path,
                is_valid=True,
                message="All validation checks passed.",
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
