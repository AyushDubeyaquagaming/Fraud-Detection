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
from fraud_detection.utils.common import ensure_dir, load_joblib, read_json, save_joblib, write_json

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ModelPusher:
    def __init__(
        self,
        config: ModelPusherConfig,
        training_artifact: ModelTrainingArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.config = config
        self.training_artifact = training_artifact
        self.evaluation_artifact = evaluation_artifact

    def _register_to_staging(
        self,
        bundle_path: Path,
        git_sha: str,
        promoted_at: str,
    ) -> dict[str, Any]:
        """Log the bundle as an MLflow artifact and register the resulting URI in
        the Model Registry under stage='Staging'.

        Always returns a status dict (never None). Filesystem promotion is the
        source of truth — registry failures are reported via the dict but do
        not block promotion. The dict's `succeeded` flag tells the caller
        whether the version was actually created.

        The bundle is logged into the currently active MLflow run when one is
        open (typical when called from TrainingPipeline). When no run is active,
        a small dedicated run is opened just for the registration so the model
        URI has a backing run id.
        """
        try:
            import mlflow
            from fraud_detection.utils.mlflow_utils import register_model_to_staging
        except Exception as exc:
            logger.exception("MLflow unavailable — skipping registry")
            return {
                "attempted": False,
                "succeeded": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_stage": "pusher_import",
            }

        try:
            active_run = mlflow.active_run()
            opened_run = False
            if active_run is None:
                mlflow.start_run(run_name=f"model_pusher_{promoted_at}")
                active_run = mlflow.active_run()
                opened_run = True

            try:
                # Log bundle so the registered version has a real artifact URI.
                mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")
                run_id = active_run.info.run_id
                model_uri = f"runs:/{run_id}/model_bundle/{bundle_path.name}"

                description = (
                    f"git_sha={git_sha} promoted_at={promoted_at} "
                    f"capture_rate_top_5pct="
                    f"{self.evaluation_artifact.combined_oos_capture_rate_top_5pct:.4f} "
                    f"lift_top_5pct={self.evaluation_artifact.combined_oos_lift_top_5pct:.2f}"
                )
                tags = {
                    "git_sha": git_sha,
                    "promoted_at": promoted_at,
                    "capture_rate_top_5pct": f"{self.evaluation_artifact.combined_oos_capture_rate_top_5pct:.4f}",
                    "lift_top_5pct": f"{self.evaluation_artifact.combined_oos_lift_top_5pct:.4f}",
                    "model_version_label": self.config.model_version,
                }

                result = register_model_to_staging(
                    artifact_uri=model_uri,
                    registered_name=self.config.registered_model_name,
                    description=description,
                    tags=tags,
                    archive_existing_staging=self.config.archive_existing_staging,
                )
                if not result.get("succeeded"):
                    # Carry the diagnostic upward — caller writes it to
                    # promotion_metadata.json so we can grep for what broke.
                    result.setdefault("model_uri", model_uri)
                    result.setdefault("run_id", run_id)
                    return result
                return {
                    "attempted": True,
                    "succeeded": True,
                    "name": str(result["name"]),
                    "version": str(result["version"]),
                    "stage": str(result["stage"]),
                    "run_id": str(result.get("run_id") or run_id),
                }
            finally:
                if opened_run:
                    try:
                        mlflow.end_run()
                    except Exception:
                        logger.exception("mlflow.end_run failed during cleanup")
        except Exception as exc:
            logger.exception("Registry registration failed (non-fatal)")
            return {
                "attempted": True,
                "succeeded": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_stage": "pusher_outer",
            }

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logger.info("ModelPusher: starting — gate_passed=%s", self.evaluation_artifact.gate_passed)
        try:
            ensure_dir(self.config.current_dir)
            run_dir = self.training_artifact.iso_forest_path.parent.parent  # run_dir/model_training → run_dir

            eval_report = read_json(self.evaluation_artifact.evaluation_report_path)
            training_report = read_json(self.training_artifact.training_report_path)
            promotion_metadata_path = self.config.current_dir / "promotion_metadata.json"

            if not self.evaluation_artifact.gate_passed:
                cap5 = self.evaluation_artifact.combined_oos_capture_rate_top_5pct
                lift5 = self.evaluation_artifact.combined_oos_lift_top_5pct
                reason_parts = []
                if cap5 < self.config.min_capture_rate_top_5pct:
                    reason_parts.append(
                        f"capture_rate_top_5pct={cap5:.3f} < {self.config.min_capture_rate_top_5pct:.3f}"
                    )
                if lift5 < self.config.min_lift_top_5pct:
                    reason_parts.append(
                        f"lift_top_5pct={lift5:.2f}x < {self.config.min_lift_top_5pct:.2f}x"
                    )
                metadata = {
                    "gate_passed": False,
                    "reason": "; ".join(reason_parts) or "gate_passed=false in evaluation",
                    "combined_oos_capture_rate_top_5pct": cap5,
                    "combined_oos_lift_top_5pct": lift5,
                    "gate_thresholds": {
                        "min_capture_rate_top_5pct": self.config.min_capture_rate_top_5pct,
                        "min_lift_top_5pct": self.config.min_lift_top_5pct,
                    },
                    "capture_rates": eval_report.get("capture_rates", {}),
                    "capture_stats": eval_report.get("capture_stats", {}),
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                    "git_sha": _git_sha(),
                }
                write_json(metadata, promotion_metadata_path)
                logger.error("ModelPusher: NOT promoted — %s", metadata["reason"])
                return ModelPusherArtifact(
                    model_bundle_path=promotion_metadata_path,
                    promotion_metadata_path=promotion_metadata_path,
                    promoted=False,
                )

            # --- Build model bundle ---
            iso_forest = load_joblib(self.training_artifact.iso_forest_path)
            kmeans = load_joblib(self.training_artifact.kmeans_path)
            mahal_stats = load_joblib(self.training_artifact.mahalanobis_stats_path)
            scalers = load_joblib(self.training_artifact.scaler_path)
            lr_models = load_joblib(self.training_artifact.lr_operational_path)

            pd = __import__("pandas")
            scored_df = pd.read_parquet(self.evaluation_artifact.scored_players_path)

            # Build frozen CCS lookup from pre-computed player features.
            # CCS columns are cohort-level stats already merged into scored_df.
            _ccs_cols = ["primary_ccs_id", "ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]
            if all(c in scored_df.columns for c in _ccs_cols):
                ccs_stats_lookup = (
                    scored_df[_ccs_cols]
                    .drop_duplicates(subset=["primary_ccs_id"])
                    .set_index("primary_ccs_id")
                )
            else:
                ccs_stats_lookup = pd.DataFrame(
                    columns=["ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]
                )

            bundle = {
                "iso_forest": iso_forest,
                "kmeans": kmeans,
                "mahal_stats": mahal_stats,
                # Top-level for hybrid_inference.py compatibility
                "mean_vec": mahal_stats["mean_vec"],
                "cov_inv": mahal_stats["cov_inv"],
                "scaler_unsup": scalers["scaler_unsup"],
                "scaler_operational": scalers["scaler_operational"],
                "style_scaler": scalers.get("style_scaler"),
                "style_pca": scalers.get("style_pca"),
                "full_pca": scalers.get("full_pca"),
                "lr_operational": lr_models["lr_operational"],
                "feature_columns": self.training_artifact.feature_columns,
                "log1p_columns": sorted(training_report.get("log1p_cols", [])),
                "style_columns": training_report.get("style_columns", []),
                "style_log1p_columns": training_report.get("style_log1p_cols", []),
                "iso_min": float(scored_df["iso_forest_score"].min()),
                "iso_max": float(scored_df["iso_forest_score"].max()),
                "mahal_min": float(scored_df["mahalanobis_dist"].min()),
                "mahal_max": float(scored_df["mahalanobis_dist"].max()),
                "cluster_min": float(scored_df["cluster_distance"].min()),
                "cluster_max": float(scored_df["cluster_distance"].max()),
                "risk_p80": float(eval_report["risk_p80"]),
                "risk_p95": float(eval_report["risk_p95"]),
                "anomaly_weight": float(eval_report["anomaly_weight"]),
                "supervised_weight": float(eval_report["supervised_weight"]),
                "anomaly_component_weights": eval_report.get("anomaly_component_weights", {}),
                "reference_size": int(len(scored_df)),
                "ccs_stats_lookup": ccs_stats_lookup,
            }
            bundle_path = self.config.current_dir / "model_bundle.joblib"
            save_joblib(bundle, bundle_path)

            # Copy scored output + alert queue + evaluation
            shutil.copy2(
                self.evaluation_artifact.scored_players_path,
                self.config.current_dir / "hybrid_scored_players.parquet",
            )
            shutil.copy2(
                self.evaluation_artifact.evaluation_report_path,
                self.config.current_dir / "hybrid_evaluation.json",
            )
            alert_src = self.evaluation_artifact.scored_players_path.parent / "alert_queue.csv"
            if alert_src.exists():
                shutil.copy2(alert_src, self.config.current_dir / "alert_queue.csv")

            # Feature pipeline config
            feat_config = {
                "feature_columns": self.training_artifact.feature_columns,
                "log1p_cols": training_report.get("log1p_cols", []),
                "style_columns": training_report.get("style_columns", []),
                "style_log1p_cols": training_report.get("style_log1p_cols", []),
            }
            write_json(feat_config, self.config.current_dir / "feature_pipeline_config.json")

            promoted_at = datetime.now(timezone.utc).isoformat()
            git_sha = _git_sha()
            metadata = {
                "gate_passed": True,
                "run_dir": str(run_dir),
                "promoted_at": promoted_at,
                "git_sha": git_sha,
                "capture_rates": eval_report.get("capture_rates", {}),
                "capture_stats": eval_report.get("capture_stats", {}),
                "combined_oos_capture_rate_top_5pct": self.evaluation_artifact.combined_oos_capture_rate_top_5pct,
                "combined_oos_lift_top_5pct": self.evaluation_artifact.combined_oos_lift_top_5pct,
                "gate_thresholds": {
                    "min_capture_rate_top_5pct": self.config.min_capture_rate_top_5pct,
                    "min_lift_top_5pct": self.config.min_lift_top_5pct,
                },
                "combined_oos_top_20pct": self.evaluation_artifact.combined_oos_top_20pct,
            }

            # --- MLflow Model Registry (best-effort, non-fatal) ---
            # Registry adds version tracking + lineage. Filesystem promotion
            # below remains the source of truth for the serving layer; if
            # registration fails, the bundle on disk is still served.
            #
            # registry_status is ALWAYS recorded in promotion_metadata when a
            # registration was attempted, including the error type/message on
            # failure — this is the diagnostic surface for "registry shows the
            # model name but no versions".
            registry_status: dict[str, Any] | None = None
            registry_info: dict[str, Any] | None = None
            if self.config.register_on_promotion:
                registry_status = self._register_to_staging(
                    bundle_path=bundle_path,
                    git_sha=git_sha,
                    promoted_at=promoted_at,
                )
                metadata["registry_status"] = registry_status
                if registry_status.get("succeeded"):
                    registry_info = {
                        "name": registry_status["name"],
                        "version": registry_status["version"],
                        "stage": registry_status["stage"],
                        "run_id": registry_status.get("run_id"),
                    }
                    metadata["mlflow_registry"] = registry_info

            write_json(metadata, promotion_metadata_path)

            # Write serving manifest atomically — only on successful promotion.
            # The API reads this file to find the immutable promoted run directory.
            run_id = run_dir.name
            serving_manifest = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "promoted_at": promoted_at,
                "git_sha": git_sha,
                "model_version": self.config.model_version,
            }
            if registry_info:
                serving_manifest["mlflow_registry"] = registry_info
            manifest_path = self.config.current_dir / self.config.manifest_file
            tmp_manifest_path = self.config.current_dir / f"{self.config.manifest_file}.tmp"
            write_json(serving_manifest, tmp_manifest_path)
            tmp_manifest_path.replace(manifest_path)

            logger.info("ModelPusher: promoted successfully to %s", self.config.current_dir)
            return ModelPusherArtifact(
                model_bundle_path=bundle_path,
                promotion_metadata_path=promotion_metadata_path,
                promoted=True,
                registered_model_name=(registry_info or {}).get("name"),
                registered_model_version=(registry_info or {}).get("version"),
                registered_model_stage=(registry_info or {}).get("stage"),
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
