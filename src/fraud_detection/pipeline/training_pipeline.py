from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.feature_engineering import FeatureEngineering
from fraud_detection.components.model_evaluation import ModelEvaluation
from fraud_detection.components.model_pusher import ModelPusher
from fraud_detection.components.model_training import ModelTraining
from fraud_detection.components.monitoring import Monitoring
from fraud_detection.constants.constants import BATCH_SCORING_CONFIG_FILE_PATH, CONFIG_FILE_PATH, REPO_ROOT
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainingConfig,
    MonitoringConfig,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, read_yaml, write_json

logger = get_logger(__name__)


def _make_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


class TrainingPipeline:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config_path = config_path

    def run(self) -> Path:
        logger.info("TrainingPipeline: starting")
        config_dict = read_yaml(self.config_path)

        run_id = _make_run_id()
        artifact_root = _resolve_repo_path(config_dict["pipeline"]["artifact_root"])
        run_dir = artifact_root / "runs" / run_id
        current_dir = _resolve_repo_path(config_dict["pipeline"]["current_dir"])
        random_seed = int(config_dict["pipeline"].get("random_seed", 42))
        ensure_dir(run_dir)

        ing_cfg = config_dict["data_ingestion"]
        val_cfg = config_dict["data_validation"]
        partnership_cfg = config_dict.get("partnership", {})
        use_candidate_store = bool(partnership_cfg.get("use_candidate_store", False))
        eval_cfg = config_dict["model_evaluation"]
        serving_cfg = config_dict.get("serving", {})
        batch_scoring_config_path = _resolve_repo_path(
            config_dict["pipeline"].get("weekly_serving_snapshot_config", BATCH_SCORING_CONFIG_FILE_PATH)
        )

        # Build component configs
        data_ingestion_config = DataIngestionConfig(
            source=ing_cfg["source"],
            parquet_path=REPO_ROOT / ing_cfg["parquet_path"],
            mongo_uri_env_var=ing_cfg["mongodb"]["uri_env_var"],
            mongo_database_env_var=ing_cfg["mongodb"]["database_env_var"],
            mongo_collection_env_var=ing_cfg["mongodb"]["collection_env_var"],
            output_dir=run_dir / "data_ingestion",
            parquet_strategy=ing_cfg.get("parquet", {}).get("strategy", "full_copy"),
            parquet_strategy_params=dict(ing_cfg.get("parquet", {}).get("strategy_params", {})),
            mongo_strategy=ing_cfg["mongodb"].get("strategy", "date_window"),
            mongo_strategy_params=dict(ing_cfg["mongodb"].get("strategy_params", {})),
        )
        data_validation_config = DataValidationConfig(
            schema_path=REPO_ROOT / "configs" / "schema.yaml",
            required_columns=val_cfg["required_columns"],
            min_row_count=int(val_cfg["min_row_count"]),
            fraud_csv_path=REPO_ROOT / val_cfg["fraud_csv_path"],
            output_dir=run_dir / "data_validation",
        )
        feature_engineering_config = FeatureEngineeringConfig(
            fraud_csv_path=REPO_ROOT / val_cfg["fraud_csv_path"],
            output_dir=run_dir / "feature_engineering",
            mode="training_eval",
            partnership=partnership_cfg,
        )
        model_training_config = ModelTrainingConfig(
            random_seed=random_seed,
            output_dir=run_dir / "model_training",
            partnership=partnership_cfg,
        )
        model_evaluation_config = ModelEvaluationConfig(
            output_dir=run_dir / "model_evaluation",
            min_capture_rate_top_5pct=float(eval_cfg.get("min_capture_rate_top_5pct", 0.40)),
            min_lift_top_5pct=float(eval_cfg.get("min_lift_top_5pct", 5.0)),
        )
        mlflow_cfg_raw = config_dict.get("mlflow", {})
        model_pusher_config = ModelPusherConfig(
            current_dir=current_dir,
            manifest_file=str(serving_cfg.get("manifest_file", "serving_manifest.json")),
            model_version=str(serving_cfg.get("model_version", "partnership_v1")),
            min_capture_rate_top_5pct=float(eval_cfg.get("min_capture_rate_top_5pct", 0.40)),
            min_lift_top_5pct=float(eval_cfg.get("min_lift_top_5pct", 5.0)),
            register_on_promotion=bool(mlflow_cfg_raw.get("register_on_promotion", True)),
            registered_model_name=str(
                mlflow_cfg_raw.get("registered_model_name", "fraud_detection_partnership_v1")
            ),
            archive_existing_staging=bool(mlflow_cfg_raw.get("archive_existing_staging", True)),
            auto_promote_to_production=bool(
                mlflow_cfg_raw.get("auto_promote_to_production", False)
            ),
        )

        mon_cfg_raw = config_dict.get("monitoring", {})
        monitoring_config = MonitoringConfig(
            enabled=bool(mon_cfg_raw.get("enabled", True)),
            reports_dir=str(mon_cfg_raw.get("reports_dir", "monitoring")),
            sample_size=int(mon_cfg_raw.get("sample_size", 50000)),
            monitored_features=list(mon_cfg_raw.get("monitored_features", [])),
            drift_threshold=float(mon_cfg_raw.get("drift_threshold", 0.3)),
            reference_from_current_metadata=bool(mon_cfg_raw.get("reference_from_current_metadata", True)),
        )

        # MLflow setup (non-fatal)
        load_dotenv(REPO_ROOT / ".env")
        experiment_name = mlflow_cfg_raw.get("experiment_name", "fraud_detection_partnership_v1")

        from fraud_detection.utils.mlflow_utils import (
            get_tracking_uri,
            log_artifact_safe,
            log_artifacts_safe,
            log_metrics_safe,
            log_params_safe,
            setup_mlflow,
        )
        import mlflow

        tracking_uri = get_tracking_uri()
        exp_id = setup_mlflow(tracking_uri, experiment_name)

        mlflow_active = False
        try:
            mlflow.start_run(run_name=run_id, experiment_id=exp_id)
            mlflow_active = True
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("source", "training_pipeline")
        except Exception as mle:
            logger.warning("MLflow run could not start: %s — continuing without MLflow", mle)

        pusher_artifact = None
        try:
            # --- Step 1: Data Ingestion ---
            if use_candidate_store:
                logger.info("[1/7] DataIngestion skipped (candidate_store mode)")
                candidate_store_path = _resolve_repo_path(partnership_cfg.get("candidate_store_path", "data_store/candidate_draws"))
                if not candidate_store_path.exists():
                    window = partnership_cfg.get("candidate_window", {})
                    raise ValueError(
                        f"candidate store missing for window {window.get('start_date')}..{window.get('end_date')}; "
                        "run scripts/extract_candidate_draws.py first"
                    )
                ingestion_report_path = run_dir / "data_ingestion" / "ingestion_report.json"
                ensure_dir(ingestion_report_path.parent)
                write_json(
                    {
                        "source_type": "candidate_store",
                        "candidate_store_path": str(candidate_store_path),
                        "date_range": partnership_cfg.get("candidate_window", {}),
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ingestion_report_path,
                )
                ingestion_artifact = DataIngestionArtifact(
                    raw_data_path=candidate_store_path,
                    ingestion_report_path=ingestion_report_path,
                    row_count=0,
                    member_count=0,
                    source_type="candidate_store",
                    strategy_used="candidate_store",
                    date_range=partnership_cfg.get("candidate_window", {}),
                )
            else:
                logger.info("[1/7] DataIngestion")
                ingestion_artifact = DataIngestion(data_ingestion_config).initiate_data_ingestion()

            # --- Step 2: Data Validation ---
            logger.info("[2/7] DataValidation")
            DataValidation(data_validation_config, ingestion_artifact).initiate_data_validation()

            # --- Step 3: Feature Engineering ---
            logger.info("[3/7] FeatureEngineering")
            fe_artifact = FeatureEngineering(
                feature_engineering_config, ingestion_artifact
            ).initiate_feature_engineering()

            if mlflow_active:
                log_params_safe({
                    "source": ing_cfg["source"],
                    "random_seed": random_seed,
                    "fraud_player_count": fe_artifact.fraud_player_count,
                    "dropped_positive_count": fe_artifact.dropped_positive_count,
                    "feature_count": len(fe_artifact.feature_columns),
                    "model_version": "partnership_v1",
                })

            # --- Step 4: Model Training ---
            logger.info("[4/7] ModelTraining")
            training_artifact = ModelTraining(
                model_training_config, fe_artifact
            ).initiate_model_training()

            if mlflow_active:
                with open(training_artifact.training_report_path) as f:
                    tr = json.load(f)
                log_metrics_safe({
                    "stage1_pr_auc": (tr.get("stage1") or {}).get("pr_auc") or 0,
                    "stage2_pr_auc": (tr.get("stage2") or {}).get("pr_auc") or 0,
                    "fraud_player_count": tr.get("fraud_members", 0),
                })
                log_artifact_safe(str(training_artifact.training_report_path))

            # --- Step 5: Model Evaluation ---
            logger.info("[5/7] ModelEvaluation")
            eval_artifact = ModelEvaluation(
                model_evaluation_config, training_artifact
            ).initiate_model_evaluation()

            if mlflow_active:
                log_metrics_safe({
                    "stage2_capture_rate_top_5pct": eval_artifact.stage2_capture_rate_top_5pct,
                    "stage2_lift_top_5pct": eval_artifact.stage2_lift_top_5pct,
                    "stage2_top_50_captured": eval_artifact.stage2_top_50_captured,
                    "gate_passed": int(eval_artifact.gate_passed),
                })
                log_artifact_safe(str(eval_artifact.evaluation_report_path))
                plots_dir = eval_artifact.evaluation_report_path.parent / "plots"
                if plots_dir.exists():
                    log_artifacts_safe(str(plots_dir))

            # --- Step 6: Monitoring (non-blocking) ---
            logger.info("[6/7] Monitoring")
            monitoring_artifact = Monitoring(
                config=monitoring_config,
                current_dir=current_dir,
                ingestion_artifact=ingestion_artifact,
                fe_artifact=fe_artifact,
                training_artifact=training_artifact,
                eval_artifact=eval_artifact,
                run_dir=run_dir,
            ).initiate_monitoring()

            if mlflow_active and monitoring_artifact.monitoring_completed and monitoring_artifact.reports_dir:
                from fraud_detection.utils.mlflow_utils import log_artifacts_safe, log_artifact_safe
                log_artifacts_safe(str(monitoring_artifact.reports_dir))
                if monitoring_artifact.drift_summary_path:
                    log_artifact_safe(str(monitoring_artifact.drift_summary_path))

            # --- Step 7: Model Pusher ---
            logger.info("[7/8] ModelPusher")
            pusher_artifact = ModelPusher(
                model_pusher_config, training_artifact, eval_artifact
            ).initiate_model_pusher()

            # --- Step 8: Weekly Serving Snapshot ---
            if pusher_artifact.promoted:
                logger.info("[8/8] WeeklyServingSnapshot")
                from fraud_detection.pipeline.batch_scoring_pipeline import BatchScoringPipeline

                BatchScoringPipeline(config_path=batch_scoring_config_path).run()
            else:
                logger.info("[8/8] WeeklyServingSnapshot skipped because promotion gate did not pass")

            if mlflow_active:
                mlflow.set_tag("promoted", "true" if pusher_artifact.promoted else "false")
                if pusher_artifact.promoted:
                    log_artifact_safe(str(pusher_artifact.model_bundle_path))
                mlflow.end_run(status="FINISHED")

        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.error("TrainingPipeline FAILED:\n%s", tb_str)
            if mlflow_active:
                try:
                    mlflow.end_run(status="FAILED")
                except Exception:
                    pass
            write_json(
                {"status": "FAILED", "error": str(exc), "traceback": tb_str},
                run_dir / "run_metadata.json",
            )
            raise FraudDetectionException(exc, sys) from exc

        write_json(
            {
                "run_id": run_id,
                "status": "FINISHED",
                "promoted": pusher_artifact.promoted if pusher_artifact else False,
                "run_dir": str(run_dir),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            run_dir / "run_metadata.json",
        )

        logger.info(
            "TrainingPipeline: complete — run_id=%s, promoted=%s",
            run_id, pusher_artifact.promoted if pusher_artifact else False,
        )
        return run_dir
