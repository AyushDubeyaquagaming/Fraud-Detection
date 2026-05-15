from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "partnership_v1",
                "source_run_id": "run_20260422_105102",
                "promoted_at": "2026-04-22T15:03:55.377091+00:00",
                "evaluated_at": "2026-04-22T15:03:55.299996+00:00",
                "snapshot_available": True,
                "snapshot_status": "ready",
                "snapshot_reason": None,
                "snapshot_lookback_days": 7,
                "stage2_alert_threshold": 0.65,
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
    total_holdout_members: int = 0
    total_evaluation_members: int = 0
    validation_status: str | None = None
    promotion_decision: str | None = None
    stage2_alert_threshold: float | None = None
    artifacts_loaded_at: str


class ReloadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "reloaded",
                "previous_run_id": "run_20260421_161503",
                "current_run_id": "run_20260422_105102",
                "reloaded_at": "2026-04-23T12:05:00+00:00",
                "total_holdout_members": 109708,
            }
        }
    )

    status: Literal["reloaded"]
    previous_run_id: str | None
    current_run_id: str
    reloaded_at: str
    total_holdout_members: int
    total_evaluation_members: int = 0


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


class DrawPlayerBet(BaseModel):
    number: int | str
    bet_amount: float


class DrawPlayer(BaseModel):
    member_id: str
    total_bet_amount: float
    bets: list[DrawPlayerBet]
    ccs_id: str | None = None
    win_points: float = 0.0


class DrawPayloadScoreRequest(BaseModel):
    draw_id: int = Field(..., ge=0)
    players: list[DrawPlayer] = Field(default_factory=list)
    trans_date: str | None = None


class PartnershipMatch(BaseModel):
    member_ids: list[str]
    stage1_score_max: float
    stage1_score_mean: float
    union_coverage: float
    jaccard: float
    per_position_ratio: float
    combined_bet_cv: float
    pair_net: float
    is_section_a: bool = False
    is_section_b: bool = False


class FlaggedMember(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    member_id: str
    stage1_score_in_draw: float
    stage2_score: float
    best_partner_member_id: str | None = None
    ccs_profit_share_1d: float | None = None
    ccs_profit_share_7d: float | None = None
    ccs_total_profit_1d: float | None = None
    ccs_total_profit_7d: float | None = None
    ccs_member_count_1d: float | None = None
    ccs_member_count_7d: float | None = None
    ccs_high_concentration_1d: float | None = None
    ccs_high_concentration_7d: float | None = None
    ccs_solo_member_1d: float | None = None
    ccs_solo_member_7d: float | None = None
    bet_amount: float | None = Field(default=None, alias="betAmount")
    win_amount: float | None = Field(default=None, alias="winAmount")
    high_amount_flag: bool = Field(default=False, alias="highAmountFlag")
    high_amount_reason: str | None = Field(default=None, alias="highAmountReason")


class DrawScoreResponse(BaseModel):
    draw_id: int
    scored_at: str
    model_version: str
    source_run_id: str | None = None
    n_members_in_draw: int
    candidate_members: list[str]
    partnerships: list[PartnershipMatch]
    flagged_members: list[FlaggedMember]
    max_stage1_score: float
    max_stage2_score: float
    requires_review: bool
    response_details: list[str] = Field(default_factory=list)


class EvidenceDraw(BaseModel):
    draw_id: int
    score_reason: str
    partner_member_ids: list[str] = Field(default_factory=list)
    stage1_score_in_draw: float | None = None
    stage2_score: float | None = None
    max_union_coverage: float | None = None


class MemberScoreResponse(BaseModel):
    member_id: str
    risk_tier: Literal["HIGH", "LOW"]
    lookback_days: int
    draws_scanned: int
    evidence_draws: list[EvidenceDraw]


class CcsScoreRequest(BaseModel):
    ccs_ids: list[str] | None = None
    lookback_days: int = Field(default=7, ge=1, le=30)
    max_draws: int = Field(default=10000, ge=1, le=50000)


class CcsScore(BaseModel):
    ccs_id: str
    risk_tier: Literal["HIGH", "LOW"]
    flagged_member_count: int
    flagged_members: list[str]
    evidence_draw_ids: list[int]
    evidence: list[dict[str, str | int | float | bool | None]] = Field(default_factory=list)


class CcsScoreResponse(BaseModel):
    lookback_days: int
    draws_scanned: int
    ccs_scores: list[CcsScore]


class AlertFlaggedMember(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    member_id: str
    stage1_score_in_draw: float
    best_partner_member_id: str | None = None
    bet_amount: float | None = Field(default=None, alias="betAmount")
    win_amount: float | None = Field(default=None, alias="winAmount")
    high_amount_flag: bool = Field(default=False, alias="highAmountFlag")
    high_amount_reason: str | None = Field(default=None, alias="highAmountReason")
    ccs_id: str = Field(alias="ccsId")
    partner_member_ids: list[str] = Field(default_factory=list)


class AlertDraw(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    draw_id: int
    draw_date: str | None = None
    risk_tier: Literal["HIGH"]
    partnership_count: int
    flagged_member_count: int
    flagged_member_ids: list[str]
    flagged_members: list[AlertFlaggedMember] = Field(default_factory=list, alias="flaggedMembers")
    ccs_ids: list[str]
    high_amount_member_count: int = Field(default=0, alias="highAmountMemberCount")
    max_bet_amount: float = Field(default=0.0, alias="maxBetAmount")
    max_win_amount: float = Field(default=0.0, alias="maxWinAmount")
    max_stage1_score: float
    response_details: list[str] = Field(default_factory=list)


class AlertDrawResponse(BaseModel):
    lookback_days: int
    draws_scanned: int
    alert_draw_count: int
    alerts: list[AlertDraw]
