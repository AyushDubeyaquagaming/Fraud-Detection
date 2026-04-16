from __future__ import annotations

import os
from typing import Any

from fraud_detection.logger import get_logger

logger = get_logger(__name__)


def get_tracking_uri() -> str:
    from dotenv import load_dotenv
    from fraud_detection.constants.constants import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    # Use file:// URI for local paths so MLflow accepts them correctly
    local_path = REPO_ROOT / "mlruns"
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path.as_uri()  # → "file:///C:/..."


def get_or_create_experiment(name: str) -> str:
    import mlflow

    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp_id = get_or_create_experiment(experiment_name)
        logger.info("MLflow tracking URI: %s | experiment: %s (id=%s)", tracking_uri, experiment_name, exp_id)
        return exp_id
    except Exception as e:
        logger.warning("MLflow setup failed: %s — falling back to local mlruns", e)
        import mlflow
        from fraud_detection.constants.constants import REPO_ROOT

        local_path = REPO_ROOT / "mlruns"
        local_path.mkdir(parents=True, exist_ok=True)
        fallback_uri = local_path.as_uri()
        try:
            mlflow.set_tracking_uri(fallback_uri)
            return get_or_create_experiment(experiment_name)
        except Exception as e2:
            logger.warning("MLflow fallback also failed: %s — MLflow logging disabled", e2)
            return "0"


def log_params_safe(params: dict[str, Any]) -> None:
    try:
        import mlflow

        flat = {str(k): str(v)[:250] for k, v in params.items()}
        mlflow.log_params(flat)
    except Exception as e:
        logger.warning("mlflow.log_params failed: %s", e)


def log_metrics_safe(metrics: dict[str, float]) -> None:
    try:
        import mlflow

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
    except Exception as e:
        logger.warning("mlflow.log_metrics failed: %s", e)


def log_artifact_safe(path: str) -> None:
    try:
        import mlflow

        mlflow.log_artifact(path)
    except Exception as e:
        logger.warning("mlflow.log_artifact failed for %s: %s", path, e)


def log_artifacts_safe(path: str) -> None:
    try:
        import mlflow

        mlflow.log_artifacts(path)
    except Exception as e:
        logger.warning("mlflow.log_artifacts failed for %s: %s", path, e)
