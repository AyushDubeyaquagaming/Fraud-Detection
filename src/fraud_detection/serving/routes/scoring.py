from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import ArtifactCache, get_cache
from ..schemas import ErrorResponse, ScoreResponse

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get(
    "/{member_id}",
    response_model=ScoreResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_member_score(
    member_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> ScoreResponse:
    normalized_id = member_id.strip().upper()

    bundle = cache.get_bundle()
    df = bundle.scored_players_df

    if normalized_id not in df.index:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found in the current promoted scoring cohort",
        )

    row = df.loc[normalized_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    ccs_id = row.get("primary_ccs_id") or row.get("ccs_id")
    if pd.isna(ccs_id) if ccs_id is not None else False:
        ccs_id = None

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
