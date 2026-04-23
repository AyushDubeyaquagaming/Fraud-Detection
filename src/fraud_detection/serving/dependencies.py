from __future__ import annotations

from threading import RLock
from typing import Optional

from fastapi import HTTPException, status

from fraud_detection.logger import get_logger

from .artifact_provider import ArtifactBundle, ArtifactProvider

logger = get_logger(__name__)


class ArtifactCache:
    def __init__(self, provider: ArtifactProvider):
        self._provider = provider
        self._bundle: Optional[ArtifactBundle] = None
        self._lock = RLock()

    def initial_load(self) -> None:
        try:
            bundle = self._provider.load()
        except Exception as exc:
            logger.warning("Initial artifact load failed: %s", exc)
            return

        with self._lock:
            self._bundle = bundle

    def reload(self) -> tuple[Optional[str], str]:
        new_bundle = self._provider.load()
        with self._lock:
            previous_run_id = self._bundle.source_run_id if self._bundle else None
            self._bundle = new_bundle
            return previous_run_id, new_bundle.source_run_id

    def is_loaded(self) -> bool:
        with self._lock:
            return self._bundle is not None

    def get_bundle(self) -> ArtifactBundle:
        with self._lock:
            if self._bundle is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Serving artifacts are not loaded. Try again shortly or contact operations.",
                )
            return self._bundle


_cache: Optional[ArtifactCache] = None


def init_cache(provider: ArtifactProvider) -> ArtifactCache:
    global _cache
    _cache = ArtifactCache(provider)
    return _cache


def get_cache() -> ArtifactCache:
    if _cache is None:
        raise RuntimeError("Artifact cache not initialized")
    return _cache
