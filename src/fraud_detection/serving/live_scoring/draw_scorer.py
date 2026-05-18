from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from fraud_detection.components.clique_scan import coerce_clique_rule_config
from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS, coerce_pair_rule_config
from fraud_detection.components.ccs_features import CCS_FEATURE_COLUMNS, attach_ccs_concentration_features_from_frame, prepare_profit_frame
from fraud_detection.components.partnership_features import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    PartnershipThresholds,
    build_live_stage2_frame,
    compute_partnership_features,
    compute_clique_rows_from_candidates,
    compute_pair_rows_from_candidates,
    ensure_stage1_schema,
    project_clique_scores_to_member_draw_rows,
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
        ccs_concentration_table: pd.DataFrame | None = None,
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
        self.clique_rules = coerce_clique_rule_config(bundle.get("clique_rules") or {})
        self.stage1_flag_threshold = float(
            bundle.get("stage1_flag_threshold", self.pair_rules.stage1_flag_threshold)
        )
        self.stage1_high_threshold = float(bundle.get("stage1_high_threshold", 0.5))
        self.stage2_alert_threshold = float(bundle.get("stage2_alert_threshold", 0.65))
        self.model_version = str(bundle.get("model_version", "partnership_v1"))
        self.partnership_table = partnership_table if partnership_table is not None else pd.DataFrame()
        self.ccs_concentration_table = ccs_concentration_table if ccs_concentration_table is not None else pd.DataFrame()
        self._prepared_ccs_profit_frame = self._prepare_serving_ccs_profit_frame(self.ccs_concentration_table)

    @staticmethod
    def _prepare_serving_ccs_profit_frame(ccs_concentration_table: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["ccs_id", "member_id", "profit_date", "daily_profit"]
        if ccs_concentration_table.empty or not set(required_columns).issubset(ccs_concentration_table.columns):
            return pd.DataFrame(columns=required_columns)
        return prepare_profit_frame(ccs_concentration_table)

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
        stage1_score_values = score_stage1(self.stage1_model, stage1_features, self.stage1_feature_columns)
        stage1_scores = stage1_features.merge(
            stage1_score_values[[column for column in ["member_id", "draw_id", "stage1_score"] if column in stage1_score_values.columns]],
            on=[column for column in ["member_id", "draw_id"] if column in stage1_features.columns],
            how="left",
        )
        stage1_scores["stage1_score"] = pd.to_numeric(stage1_scores.get("stage1_score"), errors="coerce").fillna(0.0)
        stage1_scores_with_ccs = self._attach_serving_ccs_context(stage1_scores)
        live_stage2 = build_live_stage2_frame(
            stage1_scores_with_ccs,
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )
        stage2_scores = score_stage2(self.stage2_model, live_stage2, self.stage2_feature_columns)
        scored_members = stage1_scores_with_ccs.merge(stage2_scores, on="member_id", how="left")
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
                    **self._ccs_context_payload(row),
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
        candidate_frame = pd.DataFrame([row])
        pair_events = compute_pair_rows_from_candidates(candidate_frame, rule_config=self.pair_rules, mode="inference")
        pair_events = self._score_candidate_pairs(pair_events)
        clique_events = compute_clique_rows_from_candidates(candidate_frame, rule_config=self.clique_rules)
        return self._candidate_result_from_signals(
            row,
            pair_events,
            clique_events,
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )

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
        clique_events = compute_clique_rows_from_candidates(candidate_rows, rule_config=self.clique_rules)
        if pair_events.empty:
            grouped_pairs: dict[int, pd.DataFrame] = {}
        else:
            grouped_pairs = {
                int(draw_id): group.copy()
                for draw_id, group in pair_events.groupby("draw_id", sort=False)
            }
        if clique_events.empty:
            grouped_cliques: dict[int, pd.DataFrame] = {}
        else:
            grouped_cliques = {
                int(draw_id): group.copy()
                for draw_id, group in clique_events.groupby("draw_id", sort=False)
            }
        return [
            self._candidate_result_from_signals(
                row,
                grouped_pairs.get(int(row["draw_id"]), pd.DataFrame()),
                grouped_cliques.get(int(row["draw_id"]), pd.DataFrame()),
                stage1_history=stage1_history,
                partnership_features=partnership_features,
            )
            for row in rows
        ]

    def _score_candidate_pairs(self, pair_events: pd.DataFrame) -> pd.DataFrame:
        if not pair_events.empty:
            strict_signal = pd.to_numeric(
                pair_events.get("is_strict_collusion_pattern", pair_events.get("is_strict_match", 0)),
                errors="coerce",
            ).fillna(0).astype(int)
            near_mask = strict_signal.eq(0)
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
            pair_events.loc[strict_signal.eq(1), "pair_risk_score"] = 1.0
            nearmiss_mask = pair_events["is_nearmiss"].astype(int).eq(1)
            pair_events.loc[nearmiss_mask, "pair_risk_score"] = pair_events.loc[nearmiss_mask, "pair_risk_score"].clip(
                lower=self.stage1_flag_threshold
            )
        return pair_events

    def _candidate_result_from_pairs(
        self,
        row: dict[str, Any],
        pair_events: pd.DataFrame,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> DrawScoreResult:
        return self._candidate_result_from_signals(
            row,
            pair_events,
            pd.DataFrame(),
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )

    def _candidate_result_from_signals(
        self,
        row: dict[str, Any],
        pair_events: pd.DataFrame,
        clique_events: pd.DataFrame,
        *,
        stage1_history: pd.DataFrame | None = None,
        partnership_features: pd.DataFrame | None = None,
    ) -> DrawScoreResult:
        draw_id = int(row["draw_id"])
        if pair_events.empty and clique_events.empty:
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
        candidate_frame = pd.DataFrame([row])
        pair_stage1 = project_pair_scores_to_member_draw_rows(pair_events, rolling_context=False)
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
            pair_stage1 = pair_stage1.merge(member_scores, on=["member_id", "draw_id"], how="left")
        clique_stage1 = project_clique_scores_to_member_draw_rows(
            clique_events,
            candidate_frame,
            rolling_context=False,
        )
        stage1_scores = self._combine_candidate_stage1_scores(pair_stage1, clique_stage1)
        stage1_scores["stage1_score"] = pd.to_numeric(stage1_scores.get("stage1_score"), errors="coerce").fillna(0.0)
        stage1_scores_with_ccs = self._attach_serving_ccs_context(stage1_scores)
        live_stage2 = build_live_stage2_frame(
            stage1_scores_with_ccs,
            stage1_history=stage1_history,
            partnership_features=partnership_features,
        )
        stage2_scores = score_stage2(self.stage2_model, live_stage2, self.stage2_feature_columns) if not live_stage2.empty else pd.DataFrame(columns=["member_id", "stage2_score"])
        scored_members = stage1_scores_with_ccs.merge(stage2_scores, on="member_id", how="left")
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
        partnerships = self._candidate_partnership_docs(pair_events) + self._candidate_clique_docs(clique_events)
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
                    **self._ccs_context_payload(item),
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
    def _combine_candidate_stage1_scores(*frames: pd.DataFrame) -> pd.DataFrame:
        usable = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return ensure_stage1_schema(pd.DataFrame()).assign(stage1_score=pd.Series(dtype=float))
        for frame in usable:
            if "stage1_score" not in frame.columns:
                frame["stage1_score"] = 0.0
            frame["stage1_score"] = pd.to_numeric(frame["stage1_score"], errors="coerce").fillna(0.0)
        combined = pd.concat(usable, ignore_index=True)
        combined["member_id"] = combined["member_id"].astype(str).str.strip().str.upper()
        combined["draw_id"] = pd.to_numeric(combined["draw_id"], errors="coerce").astype("Int64")
        combined = combined.dropna(subset=["draw_id"]).copy()
        if combined.empty:
            return ensure_stage1_schema(pd.DataFrame()).assign(stage1_score=pd.Series(dtype=float))
        combined["draw_id"] = combined["draw_id"].astype(int)
        for column in [
            "n_low_overlap_partners_in_draw",
            "n_strict_pairs_in_draw",
            "clique_size_estimate",
            "clique_total_stake_share",
            "clique_max_pair_rule_confidence",
            "max_strict_pair_score_today",
            "best_partner_union_coverage",
        ]:
            if column not in combined.columns:
                combined[column] = 0.0
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
        aggregate = combined.groupby(["member_id", "draw_id"], as_index=False).agg(
            _n_low_overlap_partners_in_draw=("n_low_overlap_partners_in_draw", "sum"),
            _n_strict_pairs_in_draw=("n_strict_pairs_in_draw", "sum"),
            _clique_size_estimate=("clique_size_estimate", "max"),
            _clique_total_stake_share=("clique_total_stake_share", "max"),
            _clique_max_pair_rule_confidence=("clique_max_pair_rule_confidence", "max"),
            _max_strict_pair_score_today=("max_strict_pair_score_today", "max"),
            _stage1_score=("stage1_score", "max"),
        )
        combined = combined.sort_values(
            ["member_id", "draw_id", "stage1_score", "best_partner_union_coverage"],
            ascending=[True, True, False, False],
        )
        best = combined.drop_duplicates(["member_id", "draw_id"], keep="first").merge(
            aggregate,
            on=["member_id", "draw_id"],
            how="left",
        )
        best["n_low_overlap_partners_in_draw"] = best["_n_low_overlap_partners_in_draw"].astype(int)
        best["n_strict_pairs_in_draw"] = best["_n_strict_pairs_in_draw"].astype(int)
        best["clique_size_estimate"] = best["_clique_size_estimate"]
        best["clique_total_stake_share"] = best["_clique_total_stake_share"]
        best["clique_max_pair_rule_confidence"] = best["_clique_max_pair_rule_confidence"]
        best["max_strict_pair_score_today"] = best["_max_strict_pair_score_today"]
        scores = best[["member_id", "draw_id", "_stage1_score"]].rename(columns={"_stage1_score": "stage1_score"})
        out = ensure_stage1_schema(best.drop(columns=[column for column in best.columns if column.startswith("_")], errors="ignore"))
        out = out.merge(scores, on=["member_id", "draw_id"], how="left")
        out["stage1_score"] = pd.to_numeric(out["stage1_score"], errors="coerce").fillna(0.0)
        return out

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

    def _attach_serving_ccs_context(self, stage1_scores: pd.DataFrame) -> pd.DataFrame:
        if stage1_scores.empty:
            return stage1_scores
        ccs_cfg = self.bundle.get("ccs_features", {}) or {}
        ccs_needed = bool(ccs_cfg.get("enabled", False)) or any(
            column in self.stage2_feature_columns for column in CCS_FEATURE_COLUMNS
        )
        if not ccs_needed:
            return stage1_scores
        required = {"ccs_id", "member_id", "profit_date", "daily_profit"}
        if self._prepared_ccs_profit_frame.empty or not required.issubset(self._prepared_ccs_profit_frame.columns):
            return attach_ccs_concentration_features_from_frame(
                stage1_scores,
                profit_frame=pd.DataFrame(columns=list(required)),
                windows_days=list(ccs_cfg.get("windows_days", [1, 7])),
                concentration_threshold=float(ccs_cfg.get("concentration_threshold", 0.70)),
            )
        return attach_ccs_concentration_features_from_frame(
            stage1_scores,
            profit_frame=self._prepared_ccs_profit_frame,
            windows_days=list(ccs_cfg.get("windows_days", [1, 7])),
            concentration_threshold=float(ccs_cfg.get("concentration_threshold", 0.70)),
        )

    @staticmethod
    def _amount_thresholds(bet_amounts: list[float], win_amounts: list[float]) -> tuple[float, float]:
        return HIGH_AMOUNT_THRESHOLD, HIGH_AMOUNT_THRESHOLD

    @staticmethod
    def _ccs_context_payload(row: Any) -> dict[str, float]:
        payload: dict[str, float] = {}
        for column in CCS_FEATURE_COLUMNS:
            value = pd.to_numeric(getattr(row, column, 0.0), errors="coerce")
            payload[column] = 0.0 if pd.isna(value) else float(value)
        return payload

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
            pd.to_numeric(
                pair_events.get("is_strict_collusion_pattern", pair_events.get("is_strict_match", 0)),
                errors="coerce",
            ).fillna(0).astype(int).eq(1)
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
                    "total_stake_ratio": float(row.stake_ratio),
                    "combined_bet_cv": 0.0,
                    "pair_net": float(row.pair_net),
                    "rule_confidence": float(getattr(row, "rule_confidence", 0.0)),
                    "is_section_a": bool(getattr(row, "is_nearmiss", 0)),
                    "is_section_b": bool(getattr(row, "is_strict_collusion_pattern", getattr(row, "is_strict_match", 0))),
                }
            )
        return docs

    def _candidate_clique_docs(self, clique_events: pd.DataFrame) -> list[dict[str, Any]]:
        if clique_events.empty:
            return []
        docs = []
        risk = pd.to_numeric(clique_events.get("clique_risk_score"), errors="coerce").fillna(0.0)
        strict = pd.to_numeric(clique_events.get("is_strict_clique"), errors="coerce").fillna(0).astype(int)
        nearmiss = pd.to_numeric(clique_events.get("is_nearmiss_clique"), errors="coerce").fillna(0).astype(int)
        flagged = clique_events.loc[strict.eq(1) | nearmiss.eq(1) | risk.ge(self.stage1_flag_threshold)]
        for row in flagged.itertuples(index=False):
            score = float(getattr(row, "clique_risk_score", 0.0) or 0.0)
            union_count = int(getattr(row, "union_count", 0) or 0)
            duplicate_count = int(getattr(row, "duplicate_position_count", 0) or 0)
            docs.append(
                {
                    "member_ids": [str(member) for member in getattr(row, "member_ids", [])],
                    "stage1_score_max": score,
                    "stage1_score_mean": score,
                    "union_coverage": float(getattr(row, "union_coverage_pct", 0.0) or 0.0),
                    "jaccard": float(duplicate_count / max(union_count, 1)),
                    "per_position_ratio": float(getattr(row, "avg_amount_ratio", 0.0) or 0.0),
                    "total_stake_ratio": float(getattr(row, "total_stake_ratio", 0.0) or 0.0),
                    "combined_bet_cv": float(getattr(row, "combined_bet_cv", 0.0) or 0.0),
                    "pair_net": float(getattr(row, "group_net", 0.0) or 0.0),
                    "rule_confidence": float(getattr(row, "rule_confidence", 0.0) or 0.0),
                    "is_section_a": bool(getattr(row, "is_nearmiss_clique", 0)),
                    "is_section_b": bool(getattr(row, "is_strict_clique", 0)),
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
                    "total_stake_ratio": float(getattr(row, "stake_ratio", 0.0) or 0.0),
                    "combined_bet_cv": float(row.combined_bet_cv),
                    "pair_net": float(row.pair_net),
                    "is_section_a": False,
                    "is_section_b": bool(getattr(row, "is_exact", 0)),
                }
            )
        return docs
