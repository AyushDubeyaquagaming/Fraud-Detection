from __future__ import annotations

import time
from datetime import timezone

from fastapi import APIRouter, Depends

from ..dependencies import ArtifactCache, get_cache
from ..schemas import HealthResponse, ModelInfoResponse, TierDistribution

router = APIRouter(tags=["system"])

_start_time = time.time()


def _build_model_info(cache: ArtifactCache) -> ModelInfoResponse:
    bundle = cache.get_bundle()
    df = bundle.scored_players_df
    tier_counts = df["risk_tier"].value_counts().to_dict()

    return ModelInfoResponse(
        model_version=bundle.model_version,
        source_run_id=bundle.source_run_id,
        promoted_at=bundle.promoted_at,
        evaluated_at=bundle.evaluated_at,
        total_scored_members=len(df),
        tier_distribution=TierDistribution(
            LOW=int(tier_counts.get("LOW", 0)),
            MEDIUM=int(tier_counts.get("MEDIUM", 0)),
            HIGH=int(tier_counts.get("HIGH", 0)),
        ),
        anomaly_weight=float(bundle.evaluation_metadata.get("anomaly_weight", 0.6)),
        supervised_weight=float(bundle.evaluation_metadata.get("supervised_weight", 0.4)),
        artifacts_loaded_at=bundle.loaded_at.isoformat(),
    )


@router.get("/health", response_model=HealthResponse)
def health_check(cache: ArtifactCache = Depends(get_cache)) -> HealthResponse:
    uptime = int(time.time() - _start_time)
    loaded = cache.is_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        artifacts_loaded=loaded,
        uptime_seconds=uptime,
    )


@router.get("/", response_model=ModelInfoResponse, summary="Root Model Info")
def root(cache: ArtifactCache = Depends(get_cache)) -> ModelInfoResponse:
    return _build_model_info(cache)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(cache: ArtifactCache = Depends(get_cache)) -> ModelInfoResponse:
    return _build_model_info(cache)
