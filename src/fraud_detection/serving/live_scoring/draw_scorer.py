from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS, coerce_pair_rule_config
from fraud_detection.components.partnership_features import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    PartnershipThresholds,
    build_live_stage2_frame,
    compute_partnership_features,
    compute_pair_rows_from_candidates,
    project_pair_scores_to_member_draw_rows,
)
from fraud_detection.components.partnership_modeling import score_stage1, score_stage2
from fraud_detection.utils.player_vectors import build_position_index, player_to_vectors

HIGH_AMOUNT_THRESHOLD = 10_000.0


@dataclass(frozen=True)
class DrawScoreResult:
    draw_id: int
    scored_at: str
    model_version: str
    source_run_id: str | None
    n_members_in_draw: int
    candidate_members: list[str]
    partnerships: list[dict[str, Any]]
    flagged_members: list[dict[str, Any]]
    member_scores: list[dict[str, Any]]
    max_stage1_score: float
    max_stage2_score: float
    requires_review: bool
    response_details: list[str]

    def to_mongo_doc(self) -> dict[str, Any]:
        return self.__dict__.copy()


class DrawScorer:
    def __init__(
        self,
        bundle: dict[str, Any],
        *,
        source_run_id: str | None = None,
        partnership_table: pd.DataFrame | None = None,
    ):
        self.bundle = bundle
        self.source_run_id = source_run_id
        self.stage1_model = bundle["stage1_model"]
        self.stage2_model = bundle["stage2_model"]
        self.use_candidate_store = bool(bundle.get("use_candidate_store", False))
        default_stage1_feature_columns = PAIR_FEATURE_COLUMNS if self.use_candidate_store else STAGE1_FEATURE_COLUMNS
        self.stage1_feature_columns = list(bundle.get("stage1_feature_columns", default_stage1_feature_columns))
        self.stage2_feature_columns = list(bundle.get("stage2_feature_columns", STAGE2_FEATURE_COLUMNS))
        self.thresholds = PartnershipThresholds(**{
            k: v for k, v in dict(bundle.get("candidate_thresholds", {})).items()
            if k in PartnershipThresholds.__dataclass_fields__
        })
        self.pair_rules = coerce_pair_rule_config(bundle.get("pair_rules") or {})
        self.stage1_flag_threshold = float(
            bundle.get("stage1_flag_threshold", self.pair_rules.stage1_flag_threshold)
        )
        self.stage1_high_threshold = float(bundle.get("stage1_high_threshold", 0.5))
        self.stage2_alert_threshold = float(bundle.get("stage2_alert_threshold", 0.65))
        self.model_version = str(bundle.get("model_version", "partnership_v1"))
        self.partnership_table = partnership_table if partnership_table is not None else pd.DataFrame()

    def score_draw(
        self,
        raw_draw_df: pd.DataFrame,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> DrawScoreResult:
        if raw_draw_df.empty:
            raise ValueError("Cannot score an empty draw.")
        if self.use_candidate_store:
            candidate_row = self._raw_draw_to_candidate_row(raw_draw_df)
            return self.score_candidate_draw(candidate_row, stage1_history=stage1_history, partnership_features=partnership_features)
        draw_id = int(pd.to_numeric(raw_draw_df["draw_id"], errors="coerce").dropna().iloc[0])
        stage1_features, pair_events, _ = compute_partnership_features(
            raw_draw_df,
            partnership_df=self.partnership_table,
            rolling_context=False,
            candidate_thresholds=self.thresholds,
        )
        stage1_scores = score_stage1(self.stage1_model, stage1_features, self.stage1_feature_columns)
        live_stage2 = build_live_stage2_frame(
            stage1_scores,
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )
        stage2_scores = score_stage2(self.stage2_model, live_stage2, self.stage2_feature_columns)
        scored_members = stage1_scores.merge(stage2_scores, on="member_id", how="left")
        scored_members["stage2_score"] = pd.to_numeric(scored_members["stage2_score"], errors="coerce").fillna(0.0)
        response_details = self._response_details(stage1_history, partnership_features)
        member_scores = [
            {
                "member_id": str(row.member_id),
                "draw_id": draw_id,
                "draw_date": None if pd.isna(row.draw_date) else pd.Timestamp(row.draw_date).isoformat(),
                "best_partner_member_id": None if pd.isna(row.best_partner_member_id) else str(row.best_partner_member_id),
                "stage1_score": float(row.stage1_score),
                "stage2_score": float(row.stage2_score),
            }
            for row in scored_members.itertuples(index=False)
        ]

        partnerships = self._partnership_docs(pair_events, stage1_scores)
        amount_context = self._raw_amount_context(raw_draw_df)
        flagged = scored_members.loc[
            (scored_members["stage1_score"] >= self.stage1_high_threshold)
            | (scored_members["stage2_score"] >= self.stage2_alert_threshold)
        ].copy()
        flagged_members = [
            self._enrich_flagged_member_amounts(
                {
                    "member_id": str(row.member_id),
                    "stage1_score_in_draw": float(row.stage1_score),
                    "stage2_score": float(row.stage2_score),
                    "best_partner_member_id": None if pd.isna(row.best_partner_member_id) else str(row.best_partner_member_id),
                },
                amount_context,
            )
            for row in flagged.itertuples(index=False)
        ]
        return DrawScoreResult(
            draw_id=draw_id,
            scored_at=datetime.now(timezone.utc).isoformat(),
            model_version=self.model_version,
            source_run_id=self.source_run_id,
            n_members_in_draw=int(stage1_features["member_id"].nunique()),
            candidate_members=sorted(stage1_features["member_id"].astype(str).unique().tolist()),
            partnerships=partnerships,
            flagged_members=flagged_members,
            member_scores=member_scores,
            max_stage1_score=float(scored_members["stage1_score"].max()) if not scored_members.empty else 0.0,
            max_stage2_score=float(scored_members["stage2_score"].max()) if not scored_members.empty else 0.0,
            requires_review=bool(flagged_members or partnerships),
            response_details=response_details,
        )

    def score_candidate_draw(
        self,
        candidate_row: dict[str, Any] | pd.Series,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> DrawScoreResult:
        row = candidate_row.to_dict() if isinstance(candidate_row, pd.Series) else dict(candidate_row)
        pair_events = compute_pair_rows_from_candidates(pd.DataFrame([row]), rule_config=self.pair_rules, mode="inference")
        pair_events = self._score_candidate_pairs(pair_events)
        return self._candidate_result_from_pairs(row, pair_events, stage1_history=stage1_history, partnership_features=partnership_features)

    def score_candidate_batch(
        self,
        candidate_rows: pd.DataFrame,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> list[DrawScoreResult]:
        if candidate_rows.empty:
            return []
        rows = candidate_rows.to_dict("records")
        pair_events = compute_pair_rows_from_candidates(candidate_rows, rule_config=self.pair_rules, mode="inference")
        pair_events = self._score_candidate_pairs(pair_events)
        if pair_events.empty:
            grouped_pairs: dict[int, pd.DataFrame] = {}
        else:
            grouped_pairs = {
                int(draw_id): group.copy()
                for draw_id, group in pair_events.groupby("draw_id", sort=False)
            }
        return [
            self._candidate_result_from_pairs(
                row,
                grouped_pairs.get(int(row["draw_id"]), pd.DataFrame()),
                stage1_history=stage1_history,
                partnership_features=partnership_features,
            )
            for row in rows
        ]

    def _score_candidate_pairs(self, pair_events: pd.DataFrame) -> pd.DataFrame:
        if not pair_events.empty:
            near_mask = pair_events["is_strict_match"].astype(int).eq(0)
            pair_events["stage1_model_score"] = 0.0
            if near_mask.any():
                try:
                    scored_pairs = score_stage1(self.stage1_model, pair_events.loc[near_mask], self.stage1_feature_columns)
                    pair_events.loc[near_mask, "stage1_model_score"] = scored_pairs["stage1_score"].to_numpy()
                except Exception:
                    pair_events.loc[near_mask, "stage1_model_score"] = 0.0
            pair_events["pair_risk_score"] = pd.to_numeric(
                pair_events["stage1_model_score"], errors="coerce"
            ).fillna(0.0)
            pair_events.loc[pair_events["is_strict_match"].astype(int).eq(1), "pair_risk_score"] = 1.0
        return pair_events

    def _candidate_result_from_pairs(
        self,
        row: dict[str, Any],
        pair_events: pd.DataFrame,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> DrawScoreResult:
        draw_id = int(row["draw_id"])
        if pair_events.empty:
            response_details = self._response_details(stage1_history, partnership_features)
            return DrawScoreResult(
                draw_id=draw_id,
                scored_at=datetime.now(timezone.utc).isoformat(),
                model_version=self.model_version,
                source_run_id=self.source_run_id,
                n_members_in_draw=len(row.get("member_ids", [])),
                candidate_members=sorted(str(member).upper() for member in row.get("member_ids", [])),
                partnerships=[],
                flagged_members=[],
                member_scores=[],
                max_stage1_score=0.0,
                max_stage2_score=0.0,
                requires_review=False,
                response_details=response_details,
            )
        stage1_features = project_pair_scores_to_member_draw_rows(pair_events, rolling_context=False)
        stage1_scores = stage1_features[["member_id", "draw_id", "draw_date", "best_partner_member_id"]].copy() if not stage1_features.empty else pd.DataFrame(columns=["member_id", "draw_id", "draw_date", "best_partner_member_id"])
        if not pair_events.empty:
            score_rows = []
            for item in pair_events.to_dict("records"):
                score = float(item.get("pair_risk_score") or 0.0)
                for member_col in ("member_a", "member_b"):
                    score_rows.append(
                        {
                            "member_id": str(item.get(member_col, "")).strip().upper(),
                            "draw_id": int(item.get("draw_id")),
                            "stage1_score": score,
                        }
                    )
            member_scores = pd.DataFrame(score_rows).groupby(["member_id", "draw_id"], as_index=False)["stage1_score"].max()
            stage1_scores = stage1_scores.merge(member_scores, on=["member_id", "draw_id"], how="left")
        else:
            stage1_scores["stage1_score"] = []
        stage1_scores["stage1_score"] = pd.to_numeric(stage1_scores.get("stage1_score"), errors="coerce").fillna(0.0)
        live_stage2 = build_live_stage2_frame(
            stage1_scores,
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )
        stage2_scores = score_stage2(self.stage2_model, live_stage2, self.stage2_feature_columns) if not live_stage2.empty else pd.DataFrame(columns=["member_id", "stage2_score"])
        scored_members = stage1_scores.merge(stage2_scores, on="member_id", how="left")
        scored_members["stage2_score"] = pd.to_numeric(scored_members.get("stage2_score"), errors="coerce").fillna(0.0)
        response_details = self._response_details(stage1_history, partnership_features)
        amount_context = self._candidate_amount_context(row)
        member_scores = [
            {
                "member_id": str(item.member_id),
                "draw_id": draw_id,
                "draw_date": None if pd.isna(item.draw_date) else pd.Timestamp(item.draw_date).isoformat(),
                "best_partner_member_id": None if pd.isna(item.best_partner_member_id) else str(item.best_partner_member_id),
                "stage1_score": float(item.stage1_score),
                "stage2_score": float(item.stage2_score),
            }
            for item in scored_members.itertuples(index=False)
        ]
        partnerships = self._candidate_partnership_docs(pair_events)
        flagged = scored_members.loc[
            (scored_members["stage1_score"] >= self.stage1_flag_threshold)
            | (scored_members["stage2_score"] >= self.stage2_alert_threshold)
        ].copy()
        flagged_members = [
            self._enrich_flagged_member_amounts(
                {
                    "member_id": str(item.member_id),
                    "stage1_score_in_draw": float(item.stage1_score),
                    "stage2_score": float(item.stage2_score),
                    "best_partner_member_id": None if pd.isna(item.best_partner_member_id) else str(item.best_partner_member_id),
                },
                amount_context,
            )
            for item in flagged.itertuples(index=False)
        ]
        return DrawScoreResult(
            draw_id=draw_id,
            scored_at=datetime.now(timezone.utc).isoformat(),
            model_version=self.model_version,
            source_run_id=self.source_run_id,
            n_members_in_draw=len(row.get("member_ids", [])),
            candidate_members=sorted(str(member).upper() for member in row.get("member_ids", [])),
            partnerships=partnerships,
            flagged_members=flagged_members,
            member_scores=member_scores,
            max_stage1_score=float(scored_members["stage1_score"].max()) if not scored_members.empty else 0.0,
            max_stage2_score=float(scored_members["stage2_score"].max()) if not scored_members.empty else 0.0,
            requires_review=bool(flagged_members or partnerships),
            response_details=response_details,
        )

    @staticmethod
    def _response_details(
        stage1_history: pd.DataFrame | None,
        partnership_features: pd.DataFrame | None,
    ) -> list[str]:
        if (stage1_history is None or stage1_history.empty) and (
            partnership_features is None or partnership_features.empty
        ):
            return ["no_member_history"]
        return []

    def _raw_draw_to_candidate_row(self, raw_draw_df: pd.DataFrame) -> dict[str, Any]:
        draw_id = int(pd.to_numeric(raw_draw_df["draw_id"], errors="coerce").dropna().iloc[0])
        position_index = build_position_index()
        rows = raw_draw_df.copy()
        rows["total_bet_amount"] = pd.to_numeric(rows.get("total_bet_amount", 0.0), errors="coerce").fillna(0.0)
        rows = rows.loc[rows["total_bet_amount"] >= self.pair_rules.min_total_bet_amount].copy()
        member_ids: list[str] = []
        ccs_ids: list[str | None] = []
        totals: list[float] = []
        wins: list[float] = []
        coverage: list[bytes] = []
        amounts: list[list[float]] = []
        for player in rows.to_dict("records"):
            coverage_bytes, amount_vector = player_to_vectors(player, position_index, self.pair_rules.board_size)
            member_ids.append(str(player.get("member_id", "")).strip().upper())
            ccs_value = player.get("ccs_id")
            ccs_ids.append(None if ccs_value is None else str(ccs_value))
            totals.append(float(player.get("total_bet_amount") or 0.0))
            wins.append(float(player.get("win_points") or 0.0))
            coverage.append(coverage_bytes)
            amounts.append(amount_vector)
        trans_dates = pd.to_datetime(rows.get("trans_date"), errors="coerce", utc=True) if "trans_date" in rows else pd.Series(dtype="datetime64[ns, UTC]")
        return {
            "draw_id": draw_id,
            "trans_date_min": trans_dates.min() if not trans_dates.empty else datetime.now(timezone.utc),
            "trans_date_max": trans_dates.max() if not trans_dates.empty else datetime.now(timezone.utc),
            "qualifying_player_count": len(member_ids),
            "member_ids": member_ids,
            "ccs_ids": ccs_ids,
            "total_bet_amounts": totals,
            "win_points": wins,
            "coverage_bytes": coverage,
            "amount_vector": amounts,
        }

    @staticmethod
    def _amount_thresholds(bet_amounts: list[float], win_amounts: list[float]) -> tuple[float, float]:
        return HIGH_AMOUNT_THRESHOLD, HIGH_AMOUNT_THRESHOLD

    @classmethod
    def _candidate_amount_context(cls, row: dict[str, Any]) -> dict[str, Any]:
        members = [str(value).strip().upper() for value in row.get("member_ids", [])]
        bet_amounts = [float(value or 0.0) for value in row.get("total_bet_amounts", [])]
        win_amounts = [float(value or 0.0) for value in row.get("win_points", [])]
        bet_threshold, win_threshold = cls._amount_thresholds(bet_amounts, win_amounts)
        lookup: dict[str, dict[str, float]] = {}
        for idx, member_id in enumerate(members):
            if not member_id:
                continue
            lookup[member_id] = {
                "bet_amount": bet_amounts[idx] if idx < len(bet_amounts) else 0.0,
                "win_amount": win_amounts[idx] if idx < len(win_amounts) else 0.0,
            }
        return {"lookup": lookup, "bet_threshold": bet_threshold, "win_threshold": win_threshold}

    @classmethod
    def _raw_amount_context(cls, raw_draw_df: pd.DataFrame) -> dict[str, Any]:
        if raw_draw_df.empty:
            return {"lookup": {}, "bet_threshold": 10_000.0, "win_threshold": 10_000.0}
        rows = raw_draw_df.copy()
        member_ids = rows.get("member_id", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
        bet_amounts = pd.to_numeric(rows.get("total_bet_amount", 0.0), errors="coerce").fillna(0.0)
        win_amounts = pd.to_numeric(rows.get("win_points", 0.0), errors="coerce").fillna(0.0)
        bet_threshold, win_threshold = cls._amount_thresholds(bet_amounts.tolist(), win_amounts.tolist())
        lookup = {
            member_id: {"bet_amount": float(bet), "win_amount": float(win)}
            for member_id, bet, win in zip(member_ids, bet_amounts, win_amounts)
            if member_id
        }
        return {"lookup": lookup, "bet_threshold": bet_threshold, "win_threshold": win_threshold}

    @staticmethod
    def _enrich_flagged_member_amounts(flagged_member: dict[str, Any], amount_context: dict[str, Any]) -> dict[str, Any]:
        member_id = str(flagged_member.get("member_id", "")).strip().upper()
        amounts = dict(amount_context.get("lookup", {}).get(member_id, {}))
        bet_amount = float(amounts.get("bet_amount", 0.0))
        win_amount = float(amounts.get("win_amount", 0.0))
        bet_threshold = float(amount_context.get("bet_threshold", 10_000.0))
        win_threshold = float(amount_context.get("win_threshold", 10_000.0))
        high_reasons = []
        if bet_amount >= bet_threshold:
            high_reasons.append(f"bet_amount >= {bet_threshold:.2f}")
        if win_amount >= win_threshold:
            high_reasons.append(f"win_amount >= {win_threshold:.2f}")
        high_amount = bool(high_reasons)
        flagged_member["bet_amount"] = bet_amount
        flagged_member["win_amount"] = win_amount
        flagged_member["high_amount_flag"] = high_amount
        flagged_member["high_amount_reason"] = " and ".join(high_reasons) if high_amount else None
        return flagged_member

    def _candidate_partnership_docs(self, pair_events: pd.DataFrame) -> list[dict[str, Any]]:
        if pair_events.empty:
            return []
        docs = []
        flagged = pair_events.loc[
            pair_events["is_strict_match"].astype(int).eq(1)
            | (pd.to_numeric(pair_events.get("pair_risk_score"), errors="coerce").fillna(0.0) >= self.stage1_flag_threshold)
        ]
        for row in flagged.itertuples(index=False):
            score = float(getattr(row, "pair_risk_score", 0.0))
            docs.append(
                {
                    "member_ids": [str(row.member_a), str(row.member_b)],
                    "stage1_score_max": score,
                    "stage1_score_mean": score,
                    "union_coverage": float(row.union_coverage_pct),
                    "jaccard": float(row.overlap_count / max(row.union_count, 1)),
                    "per_position_ratio": float(row.ratio_similarity),
                    "combined_bet_cv": 0.0,
                    "pair_net": float(row.pair_net),
                    "is_section_a": bool(getattr(row, "is_nearmiss", 0)),
                    "is_section_b": bool(getattr(row, "is_strict_match", 0)),
                }
            )
        return docs

    def _partnership_docs(self, pair_events: pd.DataFrame, stage1_scores: pd.DataFrame) -> list[dict[str, Any]]:
        if pair_events.empty:
            return []
        score_lookup = stage1_scores.set_index("member_id")["stage1_score"].to_dict()
        docs = []
        strict = pair_events.loc[
            (pd.to_numeric(pair_events.get("overlap_positions"), errors="coerce").fillna(999).eq(0))
            & (pd.to_numeric(pair_events.get("union_coverage"), errors="coerce").fillna(0.0) >= 0.95)
            & (
                pd.to_numeric(pair_events.get("is_exact"), errors="coerce").fillna(0).eq(1)
                | pd.to_numeric(pair_events.get("is_near"), errors="coerce").fillna(0).eq(1)
            )
        ].copy()
        for row in strict.itertuples(index=False):
            members = [str(row.member_a), str(row.member_b)]
            scores = [float(score_lookup.get(member, 0.0)) for member in members]
            docs.append(
                {
                    "member_ids": members,
                    "stage1_score_max": max(scores),
                    "stage1_score_mean": sum(scores) / len(scores),
                    "union_coverage": float(row.union_coverage),
                    "jaccard": float(row.jaccard),
                    "per_position_ratio": float(row.per_position_ratio),
                    "combined_bet_cv": float(row.combined_bet_cv),
                    "pair_net": float(row.pair_net),
                    "is_section_a": False,
                    "is_section_b": bool(getattr(row, "is_exact", 0)),
                }
            )
        return docs
