from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import ArtifactCache, get_cache
from ..schemas import ErrorResponse, InsufficientDataResponse, ScoreResponse

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get(
    "/{member_id}",
    response_model=ScoreResponse | InsufficientDataResponse,
    responses={503: {"model": ErrorResponse}},
)
def get_member_score(
    member_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> ScoreResponse | InsufficientDataResponse:
    normalized_id = member_id.strip().upper()

    bundle = cache.get_bundle()
    if not bundle.snapshot_available:
        return InsufficientDataResponse(
            status="insufficient_data",
            member_id=normalized_id,
            detail=bundle.snapshot_reason or "Not enough weekly data is currently available for this member.",
            evaluated_at=bundle.evaluated_at,
            promoted_at=bundle.promoted_at,
            source_run_id=bundle.source_run_id,
            model_version=bundle.model_version,
        )

    if normalized_id not in bundle.scored_players_df.index:
        return InsufficientDataResponse(
            status="insufficient_data",
            member_id=normalized_id,
            detail="Not enough weekly data is currently available for this member.",
            evaluated_at=bundle.evaluated_at,
            promoted_at=bundle.promoted_at,
            source_run_id=bundle.source_run_id,
            model_version=bundle.model_version,
        )

    row = bundle.scored_players_df.loc[normalized_id]
    ccs_id = row.get("primary_ccs_id") or row.get("ccs_id")

    return ScoreResponse(
        member_id=normalized_id,
        risk_score=float(row["risk_score"]),
        risk_tier=str(row["risk_tier"]),
        anomaly_score=float(row["anomaly_score"]),
        supervised_score=float(row["supervised_score"]),
        ccs_id=str(ccs_id) if ccs_id is not None else None,
        evaluated_at=bundle.evaluated_at,
        promoted_at=bundle.promoted_at,
        source_run_id=bundle.source_run_id,
        model_version=bundle.model_version,
    )
