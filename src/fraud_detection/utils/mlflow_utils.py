from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fraud_detection.logger import get_logger

logger = get_logger(__name__)


def _redact_uri_for_logs(uri: str) -> str:
    if not uri or "://" not in uri:
        return uri

    parts = urlsplit(uri)
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    if parts.username or parts.password:
        netloc = f"***:***@{hostname}{port}" if hostname else "***:***"
    else:
        netloc = parts.netloc

    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def get_tracking_uri() -> str:
    """Resolve the MLflow tracking URI.

    Order of resolution:
      1. MLFLOW_TRACKING_URI env var (e.g. http://prefect-server:5000 in compose).
      2. Local SQLite store at <repo>/mlruns/mlflow.db. SQLite is required for
         the MLflow Model Registry — file:// stores cannot host registered
         models. Falling back to file:// would silently disable registration.
    """
    from dotenv import load_dotenv
    from fraud_detection.constants.constants import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    mlruns_dir = REPO_ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    db_path = mlruns_dir / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


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
        logger.info(
            "MLflow tracking URI: %s | experiment: %s (id=%s)",
            _redact_uri_for_logs(tracking_uri),
            experiment_name,
            exp_id,
        )
        return exp_id
    except Exception as e:
        logger.warning("MLflow setup failed: %s — falling back to local sqlite mlruns", e)
        import mlflow
        from fraud_detection.constants.constants import REPO_ROOT

        local_dir = REPO_ROOT / "mlruns"
        local_dir.mkdir(parents=True, exist_ok=True)
        fallback_uri = f"sqlite:///{(local_dir / 'mlflow.db').as_posix()}"
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


def register_model_to_staging(
    artifact_uri: str,
    registered_name: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    archive_existing_staging: bool = True,
) -> dict[str, Any] | None:
    """Register an MLflow artifact as a new model version and transition it to Staging.

    Non-fatal: any exception is caught, logged, and None is returned. Filesystem
    promotion (model_bundle.joblib + serving_manifest.json) remains the source
    of truth for serving — the registry adds version tracking + lineage on top.

    Parameters
    ----------
    artifact_uri : str
        MLflow URI of the logged artifact, e.g. "runs:/<run_id>/model_bundle.joblib".
    registered_name : str
        Registered model name (e.g. "fraud_detection_hybrid").
    description : str, optional
        Human-readable note attached to the version (run id, gate metrics, etc.).
    tags : dict[str, str], optional
        Tags to attach to the version (e.g. git_sha, capture_rate_top_5pct).
    archive_existing_staging : bool, default True
        If True, all prior versions in 'Staging' are transitioned to 'Archived'
        atomically with the new version's promotion. The previously promoted
        bundle on disk is unaffected.

    Returns
    -------
    dict or None
        {'name', 'version', 'stage', 'run_id'} on success; None on failure.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        logger.warning("MLflow not available — skipping model registration: %s", exc)
        return None

    try:
        client = MlflowClient()

        # mlflow.register_model handles "create registered model if missing"
        # and "create new version" in a single call.
        mv = mlflow.register_model(
            model_uri=artifact_uri,
            name=registered_name,
            tags=tags or {},
        )

        if description:
            try:
                client.update_model_version(
                    name=registered_name,
                    version=mv.version,
                    description=description,
                )
            except Exception as exc:
                logger.warning("update_model_version description failed (non-fatal): %s", exc)

        client.transition_model_version_stage(
            name=registered_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=bool(archive_existing_staging),
        )

        logger.info(
            "Registered MLflow model %s v%s in stage=Staging (run_id=%s)",
            registered_name, mv.version, getattr(mv, "run_id", "unknown"),
        )
        return {
            "name": registered_name,
            "version": mv.version,
            "stage": "Staging",
            "run_id": getattr(mv, "run_id", None),
        }
    except Exception as exc:
        # Registry is best-effort. Filesystem promotion is the source of truth
        # for the serving layer, so a registry hiccup must NOT block promotion.
        logger.warning("Model registration failed (non-fatal): %s", exc)
        return None
