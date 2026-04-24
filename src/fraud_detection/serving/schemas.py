from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ScoreResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "member_id": "GK00100883",
                "risk_score": 0.022589909255277027,
                "risk_tier": "LOW",
                "anomaly_score": 0.020559106848467477,
                "supervised_score": 0.025636112865491348,
                "ccs_id": "CCS000872",
                "evaluated_at": "2026-04-22T15:03:55.299996+00:00",
                "promoted_at": "2026-04-22T15:03:55.377091+00:00",
                "source_run_id": "run_20260422_105102",
                "model_version": "hybrid_v1",
            }
        }
    )

    member_id: str
    risk_score: float = Field(..., ge=0.0)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    anomaly_score: float = Field(..., ge=0.0)
    supervised_score: float = Field(..., ge=0.0, le=1.0)
    ccs_id: str | None = None
    evaluated_at: str | None = None
    promoted_at: str | None = None
    source_run_id: str
    model_version: str


class InsufficientDataResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "insufficient_data",
                "member_id": "GK00999999",
                "detail": "Not enough weekly data is currently available for this member.",
                "evaluated_at": "2026-04-22T15:03:55.299996+00:00",
                "promoted_at": "2026-04-22T15:03:55.377091+00:00",
                "source_run_id": "run_20260422_105102",
                "model_version": "hybrid_v1",
            }
        }
    )

    status: Literal["insufficient_data"]
    member_id: str
    detail: str
    evaluated_at: str | None = None
    promoted_at: str | None = None
    source_run_id: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "artifacts_loaded": True,
                "uptime_seconds": 12847,
            }
        }
    )

    status: Literal["ok", "degraded"]
    artifacts_loaded: bool
    uptime_seconds: int


class TierDistribution(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LOW": 87766,
                "MEDIUM": 16456,
                "HIGH": 5486,
            }
        }
    )

    LOW: int = 0
    MEDIUM: int = 0
    HIGH: int = 0


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "hybrid_v1",
                "source_run_id": "run_20260422_105102",
                "promoted_at": "2026-04-22T15:03:55.377091+00:00",
                "evaluated_at": "2026-04-22T15:03:55.299996+00:00",
                "snapshot_available": True,
                "snapshot_status": "ready",
                "snapshot_reason": None,
                "snapshot_lookback_days": 7,
                "total_scored_members": 109708,
                "tier_distribution": {
                    "LOW": 87766,
                    "MEDIUM": 16456,
                    "HIGH": 5486,
                },
                "anomaly_weight": 0.6,
                "supervised_weight": 0.4,
                "artifacts_loaded_at": "2026-04-23T09:56:15.124755+00:00",
            }
        }
    )

    model_version: str
    source_run_id: str
    promoted_at: str | None
    evaluated_at: str | None
    snapshot_available: bool
    snapshot_status: Literal["ready", "insufficient_data"]
    snapshot_reason: str | None = None
    snapshot_lookback_days: int | None = None
    total_scored_members: int
    tier_distribution: TierDistribution
    anomaly_weight: float
    supervised_weight: float
    artifacts_loaded_at: str


class ReloadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "reloaded",
                "previous_run_id": "run_20260421_161503",
                "current_run_id": "run_20260422_105102",
                "reloaded_at": "2026-04-23T12:05:00+00:00",
                "total_scored_members": 109708,
            }
        }
    )

    status: Literal["reloaded"]
    previous_run_id: str | None
    current_run_id: str
    reloaded_at: str
    total_scored_members: int


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Member not found in the current promoted scoring cohort",
                "member_id": "GK99999999",
            }
        }
    )

    detail: str
    member_id: str | None = None


# ---------------------------------------------------------------------------
# Live / historical scoring schemas
# ---------------------------------------------------------------------------

class HistoricalScoreRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "member_id": "GK00100436",
                "start_date": "2026-02-20T00:00:00Z",
                "end_date": "2026-04-20T00:00:00Z",
            }
        }
    )

    member_id: str = Field(..., min_length=1)
    start_date: datetime
    end_date: datetime


class LiveScoreResponse(BaseModel):
    member_id: str
    risk_score: float = Field(..., ge=0.0)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    anomaly_score: float = Field(..., ge=0.0)
    supervised_score: float = Field(..., ge=0.0, le=1.0)
    ccs_id: str | None = None

    window_start: str
    window_end: str
    window_days: int
    data_sources: list[str]
    parquet_rows: int
    mongo_rows: int
    total_draws_scored: int

    scoring_baseline: str
    scored_at: str
    source_run_id: str
    model_version: str


class HistoricalScoreResponse(LiveScoreResponse):
    """Same shape as LiveScoreResponse — different endpoint tag only."""
    pass


class LiveInsufficientDataResponse(BaseModel):
    status: Literal["insufficient_data"]
    member_id: str
    detail: str
    evaluated_at: str | None = None
    promoted_at: str | None = None
    source_run_id: str
    model_version: str
    window_start: str
    window_end: str
    window_days: int
    data_sources: list[str]
    parquet_rows: int
    mongo_rows: int
    total_draws_scored: int
    scoring_baseline: str
    scored_at: str
