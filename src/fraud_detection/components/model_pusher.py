from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fraud_detection.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelPusherConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, read_json, write_json

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ModelPusher:
    def __init__(self, config: ModelPusherConfig, training_artifact: ModelTrainingArtifact, evaluation_artifact: ModelEvaluationArtifact):
        self.config = config
        self.training_artifact = training_artifact
        self.evaluation_artifact = evaluation_artifact

    def _register_to_staging(self, bundle_path: Path, git_sha: str, promoted_at: str) -> dict[str, Any]:
        try:
            import mlflow
            from fraud_detection.utils.mlflow_utils import register_model_to_staging
        except Exception as exc:
            return {"attempted": False, "succeeded": False, "error_type": type(exc).__name__, "error_message": str(exc)}

        try:
            active_run = mlflow.active_run()
            opened = False
            if active_run is None:
                mlflow.start_run(run_name=f"partnership_pusher_{promoted_at}")
                active_run = mlflow.active_run()
                opened = True
            try:
                mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")
                run_id = active_run.info.run_id
                return register_model_to_staging(
                    artifact_uri=f"runs:/{run_id}/model_bundle/{bundle_path.name}",
                    registered_name=self.config.registered_model_name,
                    description=f"partnership_v1 git_sha={git_sha} promoted_at={promoted_at}",
                    tags={"git_sha": git_sha, "promoted_at": promoted_at, "model_version_label": self.config.model_version},
                    archive_existing_staging=self.config.archive_existing_staging,
                )
            finally:
                if opened:
                    mlflow.end_run()
        except Exception as exc:
            return {"attempted": True, "succeeded": False, "error_type": type(exc).__name__, "error_message": str(exc)}

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logger.info("PartnershipModelPusher: starting gate_passed=%s", self.evaluation_artifact.gate_passed)
        try:
            ensure_dir(self.config.current_dir)
            promotion_metadata_path = self.config.current_dir / "promotion_metadata.json"
            if not self.evaluation_artifact.gate_passed:
                metadata = {
                    "gate_passed": False,
                    "stage2_capture_top_5pct": self.evaluation_artifact.stage2_capture_rate_top_5pct,
                    "stage2_lift_top_5pct": self.evaluation_artifact.stage2_lift_top_5pct,
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                    "git_sha": _git_sha(),
                }
                write_json(metadata, promotion_metadata_path)
                return ModelPusherArtifact(
                    model_bundle_path=promotion_metadata_path,
                    promotion_metadata_path=promotion_metadata_path,
                    promoted=False,
                )

            if self.training_artifact.model_bundle_path is None:
                raise ValueError("training_artifact.model_bundle_path is required.")
            promoted_at = datetime.now(timezone.utc).isoformat()
            git_sha = _git_sha()
            current_bundle = self.config.current_dir / "model_bundle.joblib"
            shutil.copy2(self.training_artifact.model_bundle_path, current_bundle)
            if self.training_artifact.stage1_model_path:
                shutil.copy2(self.training_artifact.stage1_model_path, self.config.current_dir / "stage1_model.joblib")
            if self.training_artifact.stage2_model_path:
                shutil.copy2(self.training_artifact.stage2_model_path, self.config.current_dir / "stage2_model.joblib")
            if self.training_artifact.partnership_table_path and self.training_artifact.partnership_table_path.exists():
                shutil.copy2(self.training_artifact.partnership_table_path, self.config.current_dir / "partnership_table.parquet")
            shutil.copy2(self.training_artifact.training_report_path, self.config.current_dir / "training_report.json")
            shutil.copy2(self.evaluation_artifact.evaluation_report_path, self.config.current_dir / "evaluation_report.json")
            shutil.copy2(self.evaluation_artifact.stage2_holdout_predictions_path, self.config.current_dir / "stage2_holdout_predictions.parquet")

            registry_info = None
            registry_status = None
            if self.config.register_on_promotion:
                registry_status = self._register_to_staging(current_bundle, git_sha, promoted_at)
                if registry_status.get("succeeded"):
                    registry_info = {
                        "name": registry_status["name"],
                        "version": registry_status["version"],
                        "stage": registry_status["stage"],
                        "run_id": registry_status.get("run_id"),
                    }

            run_dir = self.training_artifact.training_report_path.parent.parent
            metadata = {
                "gate_passed": True,
                "run_dir": str(run_dir),
                "promoted_at": promoted_at,
                "git_sha": git_sha,
                "stage2_capture_top_5pct": self.evaluation_artifact.stage2_capture_rate_top_5pct,
                "stage2_lift_top_5pct": self.evaluation_artifact.stage2_lift_top_5pct,
                "registry_status": registry_status,
                "mlflow_registry": registry_info,
            }
            write_json(metadata, promotion_metadata_path)
            manifest = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "promoted_at": promoted_at,
                "git_sha": git_sha,
                "model_version": self.config.model_version,
                "model_bundle_file": "model_bundle.joblib",
                "stage1_model_file": "stage1_model.joblib",
                "stage2_model_file": "stage2_model.joblib",
                "partnership_table_file": "partnership_table.parquet",
                "stage2_alert_threshold": read_json(self.training_artifact.training_report_path).get("stage2_alert_threshold", 0.65),
            }
            if registry_info:
                manifest["mlflow_registry"] = registry_info
            write_json(manifest, self.config.current_dir / self.config.manifest_file)
            return ModelPusherArtifact(
                model_bundle_path=current_bundle,
                promotion_metadata_path=promotion_metadata_path,
                promoted=True,
                registered_model_name=(registry_info or {}).get("name"),
                registered_model_version=(registry_info or {}).get("version"),
                registered_model_stage=(registry_info or {}).get("stage"),
            )
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc
