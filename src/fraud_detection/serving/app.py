from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraud_detection.constants.constants import CONFIG_FILE_PATH, REPO_ROOT
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import read_yaml

from .artifact_provider import ArtifactProvider, LocalDiskArtifactProvider
from .dependencies import init_cache
from .routes import admin, live_scoring, scoring, system

logger = get_logger(__name__)


def create_app(
    config_path: str | Path | None = None,
    provider: ArtifactProvider | None = None,
) -> FastAPI:
    resolved_config = Path(
        config_path or os.getenv("FRAUD_DETECTION_CONFIG") or CONFIG_FILE_PATH
    )
    config = read_yaml(resolved_config)
    serving_config = config.get("serving", {})

    if provider is None:
        provider_type = serving_config.get("artifact_provider", "local_disk")
        if provider_type != "local_disk":
            raise ValueError(f"Unknown artifact_provider: {provider_type}")

        current_dir = Path(serving_config.get("current_dir", "artifacts/current"))
        manifest_file = str(serving_config.get("manifest_file", "serving_manifest.json"))
        default_model_version = str(serving_config.get("model_version", "hybrid_v1"))
        if not current_dir.is_absolute():
            current_dir = REPO_ROOT / current_dir
        provider = LocalDiskArtifactProvider(
            current_dir=current_dir,
            manifest_file=manifest_file,
            default_model_version=default_model_version,
        )

    cache = init_cache(provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("API starting - attempting initial artifact load")
        cache.initial_load()
        if cache.is_loaded():
            logger.info("API ready - artifacts loaded")
        else:
            logger.warning("API started in degraded mode - no serving artifacts available")
        yield
        logger.info("API shutting down")

    app = FastAPI(
        title="Fraud Detection Scoring API",
        description="Lookup, live scoring, and historical scoring for promoted fraud models",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(system.router)
    app.include_router(scoring.router)
    app.include_router(live_scoring.router)
    app.include_router(admin.router)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
