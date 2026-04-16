from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraud_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("DataValidation: starting")
        try:
            ensure_dir(self.config.output_dir)
            df = pd.read_parquet(self.ingestion_artifact.raw_data_path)
            checks: dict[str, dict] = {}

            # 1. Row count
            checks["row_count"] = {
                "passed": len(df) >= self.config.min_row_count,
                "actual": len(df),
                "minimum": self.config.min_row_count,
            }

            # 2. Required columns present
            missing_cols = [c for c in self.config.required_columns if c not in df.columns]
            checks["required_columns"] = {
                "passed": len(missing_cols) == 0,
                "missing": missing_cols,
            }

            # 3. member_id not null
            if "member_id" in df.columns:
                null_members = int(df["member_id"].isna().sum())
                checks["member_id_not_null"] = {"passed": null_members == 0, "null_count": null_members}
            else:
                checks["member_id_not_null"] = {"passed": False, "null_count": "column missing"}

            # 4. draw_id not null
            if "draw_id" in df.columns:
                null_draws = int(df["draw_id"].isna().sum())
                checks["draw_id_not_null"] = {"passed": null_draws == 0, "null_count": null_draws}
            else:
                checks["draw_id_not_null"] = {"passed": False, "null_count": "column missing"}

            # 5. bets column parseable (sample 100 rows)
            if "bets" in df.columns:
                sample = df["bets"].dropna().head(100)
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
            _ts_candidates = [
                "createdAt.$date", "trans_date.$date", "updatedAt.$date",
                "createdAt", "trans_date", "updatedAt", "ts",
            ]
            ts_col = next((c for c in _ts_candidates if c in df.columns), None)
            if ts_col:
                try:
                    raw_series = df[ts_col]
                    # For .$date columns the values may be epoch ms ints or ISO strings
                    parsed = pd.to_datetime(raw_series, errors="coerce", utc=True)
                    null_ts = int(parsed.isna().sum())
                    checks["timestamp_parseable"] = {
                        "passed": null_ts < len(df),
                        "column": ts_col,
                        "null_count": null_ts,
                    }
                except Exception as ts_ex:
                    checks["timestamp_parseable"] = {"passed": False, "error": str(ts_ex)}
            else:
                # No timestamp column at all — warn but do not fail (operational data may lack it)
                checks["timestamp_parseable"] = {"passed": True, "warning": "no timestamp column found — skipped"}

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
