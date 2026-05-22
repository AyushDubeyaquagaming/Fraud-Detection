from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

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


class _LineageOnlyPyFunc:
    def predict(self, model_input, params=None):
        raise RuntimeError(
            "This MLflow registry model is lineage-only. Use the promoted local artifacts for serving and scoring."
        )


def _load_pyfunc(data_path: str):
    return _LineageOnlyPyFunc()


def log_lineage_bundle_model(bundle_path: str | Path, artifact_path: str = "model_bundle") -> str:
    import mlflow

    active_run = mlflow.active_run()
    if active_run is None:
        raise ValueError("An active MLflow run is required to log the lineage bundle model.")

    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    # Use generic artifact logging instead of mlflow.pyfunc.log_model. Newer
    # MLflow clients call the /logged-models endpoint during pyfunc logging,
    # but the compose registry server is pinned to MLflow 2.10.2 and does not
    # expose that endpoint. A hand-written MLmodel keeps registry compatibility
    # without making serving depend on MLflow.
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_path, artifacts_dir / "model_bundle.joblib")
        mlmodel = {
            "artifact_path": artifact_path,
            "flavors": {
                "python_function": {
                    "data": "artifacts",
                    "loader_module": "fraud_detection.utils.mlflow_utils",
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                }
            },
            "mlflow_version": getattr(mlflow, "__version__", "unknown"),
            "model_uuid": uuid.uuid4().hex,
            "run_id": active_run.info.run_id,
            "utc_time_created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
        (model_dir / "MLmodel").write_text(yaml.safe_dump(mlmodel, sort_keys=False), encoding="utf-8")
        mlflow.log_artifacts(str(model_dir), artifact_path=artifact_path)
    return f"runs:/{active_run.info.run_id}/{artifact_path}"


def register_model_to_staging(
    artifact_uri: str,
    registered_name: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    archive_existing_staging: bool = True,
) -> dict[str, Any]:
    """Register an MLflow artifact as a new model version and transition it to Staging.

    Non-fatal: any exception is caught and reported in the returned dict.
    Filesystem promotion (model_bundle.joblib + serving_manifest.json) remains
    the source of truth for serving — the registry adds version tracking +
    lineage on top.

    Returns
    -------
    dict
        Always returns a dict; never None. Keys:
          - attempted: bool — whether the call ran at all
          - succeeded: bool
          - on success: name, version, stage, run_id
          - on failure: error_type, error_message
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        logger.exception("MLflow not available — skipping model registration")
        return {
            "attempted": False,
            "succeeded": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_stage": "import",
        }

    try:
        client = MlflowClient()

        # mlflow.register_model handles "create registered model if missing"
        # and "create new version" in a single call.
        try:
            mv = mlflow.register_model(
                model_uri=artifact_uri,
                name=registered_name,
                tags=tags or {},
            )
        except TypeError:
            mv = mlflow.register_model(
                model_uri=artifact_uri,
                name=registered_name,
            )
            for key, value in (tags or {}).items():
                client.set_model_version_tag(
                    name=registered_name,
                    version=mv.version,
                    key=str(key),
                    value=str(value),
                )

        if description:
            try:
                client.update_model_version(
                    name=registered_name,
                    version=mv.version,
                    description=description,
                )
            except Exception:
                logger.exception("update_model_version description failed (non-fatal)")

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
            "attempted": True,
            "succeeded": True,
            "name": registered_name,
            "version": mv.version,
            "stage": "Staging",
            "run_id": getattr(mv, "run_id", None),
        }
    except Exception as exc:
        # Registry is best-effort. Filesystem promotion is the source of truth
        # for the serving layer, so a registry hiccup must NOT block promotion.
        # Use logger.exception so the full traceback hits stderr/log files,
        # then surface the error type + message in the return dict so the
        # caller can persist it to promotion_metadata.json.
        logger.exception("Model registration failed (non-fatal)")
        return {
            "attempted": True,
            "succeeded": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_stage": "register_or_transition",
            "artifact_uri": artifact_uri,
            "registered_name": registered_name,
        }
