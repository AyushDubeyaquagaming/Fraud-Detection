from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from fraud_detection.logger import get_logger

from ..dependencies import ArtifactCache, get_cache
from ..schemas import ReloadResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reload", response_model=ReloadResponse)
def reload_artifacts(cache: ArtifactCache = Depends(get_cache)) -> ReloadResponse:
    try:
        previous_run_id, current_run_id = cache.reload()
    except Exception as exc:
        logger.warning("Artifact reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reload failed. Check server logs.",
        )

    bundle = cache.get_bundle()
    return ReloadResponse(
        status="reloaded",
        previous_run_id=previous_run_id,
        current_run_id=current_run_id,
        reloaded_at=datetime.now(timezone.utc).isoformat(),
        total_scored_members=len(bundle.scored_players_df),
    )
