from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import ArtifactCache, LiveScoringContext, get_cache, get_live_scoring_context
from ..schemas import (
    ErrorResponse,
    HistoricalScoreRequest,
    HistoricalScoreResponse,
    LiveInsufficientDataResponse,
    LiveScoreResponse,
)
from ..live_scoring.feature_builder import FeatureBuilder
from ..live_scoring.scorer import LiveScorer
from ..live_scoring.window_resolver import MongoWindowFetchError, WindowResolver

router = APIRouter(prefix="/score", tags=["live_scoring"])

DEFAULT_LIVE_LOOKBACK_DAYS = 7


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _window_days(start_date: datetime, end_date: datetime) -> int:
    return max(1, int((end_date - start_date).total_seconds() // 86400))


def _scoring_baseline(window_data) -> str:
    start = window_data.training_parquet_start_date
    end = window_data.training_parquet_end_date
    if start is not None and end is not None:
        return f"Frozen training cohort from {start.date().isoformat()} to {end.date().isoformat()}"
    return "Frozen training cohort from the currently promoted run"


@router.get(
    "/live/{member_id}",
    response_model=LiveScoreResponse | LiveInsufficientDataResponse,
    responses={503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
def score_member_live(
    member_id: str,
    cache: ArtifactCache = Depends(get_cache),
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> LiveScoreResponse | LiveInsufficientDataResponse:
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=DEFAULT_LIVE_LOOKBACK_DAYS)
    normalized_id = member_id.strip().upper()
    return _score_window(
        member_id=normalized_id,
        start_date=start_utc,
        end_date=end_utc,
        cache=cache,
        context=context,
    )


@router.post(
    "/historical",
    response_model=HistoricalScoreResponse | LiveInsufficientDataResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
def score_member_historical(
    request: HistoricalScoreRequest,
    cache: ArtifactCache = Depends(get_cache),
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> HistoricalScoreResponse | LiveInsufficientDataResponse:
    start_utc = _to_utc(request.start_date)
    end_utc = _to_utc(request.end_date)
    now_utc = datetime.now(timezone.utc)

    if start_utc >= end_utc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be earlier than end_date",
        )
    if end_utc > now_utc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_date cannot be in the future",
        )
    if (end_utc - start_utc) < timedelta(days=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Date range must span at least 1 day",
        )

    normalized_id = request.member_id.strip().upper()
    return _score_window(
        member_id=normalized_id,
        start_date=start_utc,
        end_date=end_utc,
        cache=cache,
        context=context,
    )


def _score_window(
    member_id: str,
    start_date: datetime,
    end_date: datetime,
    cache: ArtifactCache,
    context: LiveScoringContext,
) -> LiveScoreResponse | LiveInsufficientDataResponse:
    bundle = cache.get_bundle()
    resolver = WindowResolver(
        training_parquet_path=context.training_raw_parquet_path,
        timestamp_field=context.timestamp_field,
        timestamp_candidates=context.timestamp_candidates,
        parquet_start_date=context.parquet_start_date,
        parquet_end_date=context.parquet_end_date,
    )
    feature_builder = FeatureBuilder(
        feature_columns=context.model_bundle["feature_columns"],
        ccs_stats_lookup=context.model_bundle["ccs_stats_lookup"],
    )
    scorer = LiveScorer(model_bundle=context.model_bundle)

    try:
        window_data = resolver.resolve(member_id, start_date, end_date)
    except MongoWindowFetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc

    scored_at = datetime.now(timezone.utc).isoformat()
    baseline = _scoring_baseline(window_data)
    total_draws = int(len(window_data.raw_df))
    if window_data.raw_df.empty:
        return LiveInsufficientDataResponse(
            status="insufficient_data",
            member_id=member_id,
            detail="No member activity was found in the requested window.",
            evaluated_at=bundle.evaluated_at,
            promoted_at=bundle.promoted_at,
            source_run_id=bundle.source_run_id,
            model_version=bundle.model_version,
            window_start=start_date.isoformat(),
            window_end=end_date.isoformat(),
            window_days=_window_days(start_date, end_date),
            data_sources=window_data.data_sources,
            parquet_rows=window_data.parquet_rows,
            mongo_rows=window_data.mongo_rows,
            total_draws_scored=0,
            scoring_baseline=baseline,
            scored_at=scored_at,
        )

    feature_df = feature_builder.build(member_id, window_data.raw_df)
    if feature_df.empty:
        return LiveInsufficientDataResponse(
            status="insufficient_data",
            member_id=member_id,
            detail="Not enough data was available to build a feature vector for this member.",
            evaluated_at=bundle.evaluated_at,
            promoted_at=bundle.promoted_at,
            source_run_id=bundle.source_run_id,
            model_version=bundle.model_version,
            window_start=start_date.isoformat(),
            window_end=end_date.isoformat(),
            window_days=_window_days(start_date, end_date),
            data_sources=window_data.data_sources,
            parquet_rows=window_data.parquet_rows,
            mongo_rows=window_data.mongo_rows,
            total_draws_scored=total_draws,
            scoring_baseline=baseline,
            scored_at=scored_at,
        )

    result = scorer.score(feature_df)
    return LiveScoreResponse(
        member_id=result.member_id,
        risk_score=result.risk_score,
        risk_tier=result.risk_tier,
        anomaly_score=result.anomaly_score,
        supervised_score=result.supervised_score,
        ccs_id=result.ccs_id,
        window_start=start_date.isoformat(),
        window_end=end_date.isoformat(),
        window_days=_window_days(start_date, end_date),
        data_sources=window_data.data_sources,
        parquet_rows=window_data.parquet_rows,
        mongo_rows=window_data.mongo_rows,
        total_draws_scored=total_draws,
        scoring_baseline=baseline,
        scored_at=scored_at,
        source_run_id=bundle.source_run_id,
        model_version=bundle.model_version,
    )
