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
from fraud_detection.constants.constants import (
    CONFIG_FILE_PATH,
    MODEL_PARAMS_FILE_PATH,
    REPO_ROOT,
)
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
        model_params = read_yaml(MODEL_PARAMS_FILE_PATH)

        run_id = _make_run_id()
        artifact_root = _resolve_repo_path(config_dict["pipeline"]["artifact_root"])
        run_dir = artifact_root / "runs" / run_id
        current_dir = _resolve_repo_path(config_dict["pipeline"]["current_dir"])
        random_seed = int(config_dict["pipeline"].get("random_seed", 42))
        ensure_dir(run_dir)

        ing_cfg = config_dict["data_ingestion"]
        val_cfg = config_dict["data_validation"]
        fe_cfg = config_dict["feature_engineering"]
        eval_cfg = config_dict["model_evaluation"]
        mp_cfg = model_params

        # Build component configs
        iso_params = dict(mp_cfg["isolation_forest"])
        iso_params["_log1p_cols"] = fe_cfg["log1p_cols"]

        data_ingestion_config = DataIngestionConfig(
            source=ing_cfg["source"],
            parquet_path=REPO_ROOT / ing_cfg["parquet_path"],
            mongo_uri_env_var=ing_cfg["mongodb"]["uri_env_var"],
            mongo_database_env_var=ing_cfg["mongodb"]["database_env_var"],
            mongo_collection_env_var=ing_cfg["mongodb"]["collection_env_var"],
            output_dir=run_dir / "data_ingestion",
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
            exclude_cols=fe_cfg["exclude_cols"],
            log1p_cols=fe_cfg["log1p_cols"],
            apply_pre_fraud_cutoff=bool(fe_cfg["apply_pre_fraud_cutoff"]),
            fraud_csv_path=REPO_ROOT / val_cfg["fraud_csv_path"],
            output_dir=run_dir / "feature_engineering",
            mode="training_eval",
        )
        model_training_config = ModelTrainingConfig(
            iso_forest_params=iso_params,
            kmeans_params=dict(mp_cfg["kmeans"]),
            lr_params=dict(mp_cfg["logistic_regression"]),
            anomaly_weight=float(mp_cfg["scoring"]["anomaly_weight"]),
            supervised_weight=float(mp_cfg["scoring"]["supervised_weight"]),
            random_seed=random_seed,
            output_dir=run_dir / "model_training",
        )
        model_evaluation_config = ModelEvaluationConfig(
            threshold_percentiles=eval_cfg["threshold_percentiles"],
            risk_tier_p80=float(eval_cfg["risk_tier_p80"]),
            risk_tier_p95=float(eval_cfg["risk_tier_p95"]),
            output_dir=run_dir / "model_evaluation",
            min_capture_rate_top_5pct=float(eval_cfg.get("min_capture_rate_top_5pct", 0.40)),
            min_lift_top_5pct=float(eval_cfg.get("min_lift_top_5pct", 5.0)),
            threshold_fixed_counts=list(eval_cfg.get("threshold_fixed_counts", [50, 500])),
            alert_queue_size=int(eval_cfg.get("alert_queue_size", 50)),
            min_capture_top_20pct=int(eval_cfg.get("min_capture_top_20pct", 0)),
        )
        model_pusher_config = ModelPusherConfig(
            current_dir=current_dir,
            min_capture_rate_top_5pct=float(eval_cfg.get("min_capture_rate_top_5pct", 0.40)),
            min_lift_top_5pct=float(eval_cfg.get("min_lift_top_5pct", 5.0)),
            min_capture_top_20pct=int(eval_cfg.get("min_capture_top_20pct", 0)),
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
        mlflow_cfg = config_dict.get("mlflow", {})
        experiment_name = mlflow_cfg.get("experiment_name", "fraud_detection_hybrid")

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
                    "iso_n_estimators": mp_cfg["isolation_forest"]["n_estimators"],
                    "iso_contamination": mp_cfg["isolation_forest"]["contamination"],
                    "kmeans_n_clusters": mp_cfg["kmeans"]["n_clusters"],
                    "lr_C": mp_cfg["logistic_regression"]["C"],
                    "anomaly_weight": mp_cfg["scoring"]["anomaly_weight"],
                    "supervised_weight": mp_cfg["scoring"]["supervised_weight"],
                    "fraud_player_count": fe_artifact.fraud_player_count,
                    "dropped_positive_count": fe_artifact.dropped_positive_count,
                    "feature_count": len(fe_artifact.feature_columns),
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
                    "pr_auc": tr.get("pr_auc", 0),
                    "roc_auc": tr.get("roc_auc", 0),
                    "fraud_player_count": tr.get("fraud_players", 0),
                })
                log_artifact_safe(str(training_artifact.training_report_path))

            # --- Step 5: Model Evaluation ---
            logger.info("[5/7] ModelEvaluation")
            eval_artifact = ModelEvaluation(
                model_evaluation_config, training_artifact
            ).initiate_model_evaluation()

            if mlflow_active:
                with open(eval_artifact.evaluation_report_path) as f:
                    ev = json.load(f)
                capture = ev.get("capture_rates", {}).get("combined_oos", {})
                log_metrics_safe({
                    "combined_oos_top_1pct": capture.get("top_1pct", 0),
                    "combined_oos_top_5pct": capture.get("top_5pct", 0),
                    "combined_oos_top_10pct": capture.get("top_10pct", 0),
                    "combined_oos_top_20pct": capture.get("top_20pct", 0),
                    "combined_oos_capture_rate_top_5pct": eval_artifact.combined_oos_capture_rate_top_5pct,
                    "combined_oos_lift_top_5pct": eval_artifact.combined_oos_lift_top_5pct,
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
                eval_artifact=eval_artifact,
                run_dir=run_dir,
            ).initiate_monitoring()

            if mlflow_active and monitoring_artifact.monitoring_completed and monitoring_artifact.reports_dir:
                from fraud_detection.utils.mlflow_utils import log_artifacts_safe, log_artifact_safe
                log_artifacts_safe(str(monitoring_artifact.reports_dir))
                if monitoring_artifact.drift_summary_path:
                    log_artifact_safe(str(monitoring_artifact.drift_summary_path))

            # --- Step 7: Model Pusher ---
            logger.info("[7/7] ModelPusher")
            pusher_artifact = ModelPusher(
                model_pusher_config, training_artifact, eval_artifact
            ).initiate_model_pusher()

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
