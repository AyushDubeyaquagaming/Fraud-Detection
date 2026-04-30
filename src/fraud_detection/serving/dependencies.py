from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from fastapi import Depends, HTTPException, status

from fraud_detection.components.feature_engineering import TIMESTAMP_CANDIDATES
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


@dataclass(frozen=True)
class LiveScoringContext:
    model_bundle: dict[str, Any]
    training_raw_parquet_path: Path
    timestamp_field: str
    timestamp_candidates: tuple[str, ...]
    parquet_start_date: Any  # datetime | None from bundle
    parquet_end_date: Any    # datetime | None from bundle


def get_live_scoring_context(
    cache: ArtifactCache = Depends(get_cache),
) -> LiveScoringContext:
    """Return the loaded model bundle and promoted raw parquet path for live scoring.

    Raises 503 if promoted artifacts are missing or incomplete.
    """
    bundle = cache.get_bundle()

    if bundle.model_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promoted model bundle is not loaded.",
        )
    if "ccs_stats_lookup" not in bundle.model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promoted model bundle is not compatible with live scoring. Re-promote artifacts.",
        )
    if bundle.training_raw_parquet_path is None or not bundle.training_raw_parquet_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promoted training raw parquet is not available.",
        )

    timestamp_field = str(bundle.snapshot_metadata.get("timestamp_field", "trans_date"))

    return LiveScoringContext(
        model_bundle=bundle.model_bundle,
        training_raw_parquet_path=bundle.training_raw_parquet_path,
        timestamp_field=timestamp_field,
        timestamp_candidates=tuple(TIMESTAMP_CANDIDATES),
        parquet_start_date=bundle.training_parquet_start_date,
        parquet_end_date=bundle.training_parquet_end_date,
    )
