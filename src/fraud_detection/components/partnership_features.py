from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.components.clique_scan import (
    CliqueRuleConfig,
    clique_rows_to_frame,
    coerce_clique_rule_config,
    emit_clique_rows,
)
from fraud_detection.components.pair_scan import (
    PAIR_FEATURE_COLUMNS,
    PairRuleConfig,
    coerce_pair_rule_config,
    emit_pair_rows,
    pair_rows_to_frame,
)

ROULETTE_POSITIONS = ["0", "00", *[str(number) for number in range(1, 37)]]


@dataclass(frozen=True)
class PartnershipThresholds:
    min_position_count: int = 4
    max_position_count: int = 34
    max_candidates_per_draw: int = 80
    max_jaccard: float = 0.10
    expandable_pair_min_coverage: float = 0.50
    section_b_min_stake: float = 1000.0
    section_b_min_per_position_ratio: float = 0.80
    section_b_max_combined_bet_cv: float = 0.35
    section_b_min_pair_net: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STAGE1_FEATURE_COLUMNS = [
    "position_count",
    "stake",
    "win_points",
    "net_result",
    "partition_score",
    "best_partner_union_coverage",
    "best_partner_jaccard",
    "best_partner_per_position_ratio",
    "best_partner_combined_bet_cv",
    "best_partner_pair_net",
    "best_partner_overlap_positions",
    "best_partner_union_positions",
    "is_exact_complementary_pair_in_draw",
    "is_near_complementary_pair_in_draw",
    "n_low_overlap_partners_in_draw",
    "n_strict_pairs_in_draw",
    "clique_size_estimate",
    "clique_total_stake_share",
    "clique_max_pair_rule_confidence",
    "max_strict_pair_score_today",
    "recurrence_1d",
    "recurrence_3d",
    "distinct_partners_1d",
    "distinct_partners_3d",
    "rolling_partnership_recurrence_7d",
    "rolling_distinct_partners_7d",
    "rolling_min_jaccard_7d",
    "rolling_pct_draws_with_complementary_pair_7d",
]

SECTION_A_FEATURE_COLUMNS = [
    "best_partnership_union_coverage",
    "max_partnership_recurrence",
    "min_partner_jaccard",
    "pct_draws_in_persistent_partnership",
    "distinct_persistent_partner_count",
    "persistent_partnership_count",
    "best_partnership_size",
    "partnership_stake_share",
]

STAGE2_FEATURE_COLUMNS = [
    "max_stage1_score",
    "mean_stage1_score",
    "n_draws_stage1_above_0p5",
    "n_draws_stage1_above_0p9",
    "longest_consecutive_high_stage1_streak",
    "n_distinct_high_score_partners",
    "max_strict_pair_score_today",
    "n_strict_pairs_in_draw",
    "clique_size_estimate",
    "clique_total_stake_share",
    "clique_max_pair_rule_confidence",
    "recurrence_1d",
    "recurrence_3d",
    "distinct_partners_1d",
    "distinct_partners_3d",
    *SECTION_A_FEATURE_COLUMNS,
    "ccs_profit_share_1d",
    "ccs_profit_share_7d",
    "ccs_total_profit_1d",
    "ccs_total_profit_7d",
    "ccs_member_count_1d",
    "ccs_member_count_7d",
    "ccs_high_concentration_1d",
    "ccs_high_concentration_7d",
    "ccs_solo_member_1d",
    "ccs_solo_member_7d",
]

PAIR_STAGE1_FEATURE_COLUMNS = PAIR_FEATURE_COLUMNS


def compute_partnership_features(
    raw_df: pd.DataFrame,
    *,
    partnership_df: pd.DataFrame | None = None,
    rolling_context: bool = True,
    candidate_only: bool = False,
    candidate_thresholds: PartnershipThresholds | dict[str, Any] | None = None,
    timestamp_field: str = "trans_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Stage 1 partnership features and pair events from raw draw rows."""
    return compute_partnership_features_legacy_rawrow(
        raw_df,
        partnership_df=partnership_df,
        rolling_context=rolling_context,
        candidate_only=candidate_only,
        candidate_thresholds=candidate_thresholds,
        timestamp_field=timestamp_field,
    )


def compute_partnership_features_legacy_rawrow(
    raw_df: pd.DataFrame,
    *,
    partnership_df: pd.DataFrame | None = None,
    rolling_context: bool = True,
    candidate_only: bool = False,
    candidate_thresholds: PartnershipThresholds | dict[str, Any] | None = None,
    timestamp_field: str = "trans_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy raw-row path kept as a fallback during candidate-store integration."""
    thresholds = _coerce_thresholds(candidate_thresholds)
    if raw_df.empty:
        return _empty_stage1_frame(), pd.DataFrame(), _coerce_partnership_df(partnership_df)

    partnership_table = _coerce_partnership_df(partnership_df)
    player_draw_vectors = build_player_draw_vectors(raw_df, timestamp_field=timestamp_field)
    stage1_features, pair_events = build_stage1_features(
        player_draw_vectors,
        partnership_table,
        min_position_count=thresholds.min_position_count,
        max_position_count=thresholds.max_position_count,
        max_candidates_per_draw=thresholds.max_candidates_per_draw,
        max_jaccard=thresholds.max_jaccard,
        expandable_pair_min_coverage=thresholds.expandable_pair_min_coverage,
        section_b_min_stake=thresholds.section_b_min_stake,
        section_b_min_per_position_ratio=thresholds.section_b_min_per_position_ratio,
        section_b_max_combined_bet_cv=thresholds.section_b_max_combined_bet_cv,
        section_b_min_pair_net=thresholds.section_b_min_pair_net,
        include_rolling_context=False,
        candidate_only=candidate_only,
    )
    if rolling_context:
        stage1_features = add_rolling_context(stage1_features, partnership_table)
    stage1_features = ensure_stage1_schema(stage1_features)
    return stage1_features, pair_events, partnership_table


def compute_pair_rows_from_candidates(
    candidate_df: pd.DataFrame,
    *,
    rule_config: PairRuleConfig | dict[str, Any] | None = None,
    mode: str = "inference",
    ordinary_negative_sample: int = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    cfg = coerce_pair_rule_config(rule_config)
    rows: list[dict[str, Any]] = []
    if candidate_df.empty:
        return pair_rows_to_frame(rows)
    for index, row in enumerate(candidate_df.itertuples(index=False)):
        row_dict = row._asdict()
        try:
            seed_offset = int(row_dict.get("draw_id"))
        except (TypeError, ValueError):
            seed_offset = index
        rows.extend(
            emit_pair_rows(
                row_dict,
                cfg,
                mode=mode,
                ordinary_negative_sample=ordinary_negative_sample,
                random_seed=random_seed + (seed_offset % 2_147_483_647),
            )
        )
    return pair_rows_to_frame(rows)


def compute_clique_rows_from_candidates(
    candidate_df: pd.DataFrame,
    *,
    rule_config: CliqueRuleConfig | dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = coerce_clique_rule_config(rule_config)
    rows: list[dict[str, Any]] = []
    if candidate_df.empty:
        return clique_rows_to_frame(rows)
    for row in candidate_df.itertuples(index=False):
        rows.extend(emit_clique_rows(row._asdict(), cfg))
    return clique_rows_to_frame(rows)


def project_pair_scores_to_member_draw_rows(pair_df: pd.DataFrame, *, rolling_context: bool = True) -> pd.DataFrame:
    if pair_df.empty:
        return ensure_stage1_schema(pd.DataFrame())

    pair_frame = pair_df.copy()
    if "pair_risk_score" not in pair_frame.columns:
        pair_frame["pair_risk_score"] = pair_frame["is_strict_match"].astype(float)
    pair_frame["pair_risk_score"] = pd.to_numeric(pair_frame["pair_risk_score"], errors="coerce").fillna(0.0)
    pair_frame["draw_date"] = pd.to_datetime(pair_frame.get("draw_date"), errors="coerce", utc=True)
    clique_lookup = _strict_clique_lookup(pair_frame)

    projected: list[dict[str, Any]] = []
    for row in pair_frame.to_dict("records"):
        for member_col, partner_col, stake_col, win_col, coverage_col in [
            ("member_a", "member_b", "stake_a", "win_points_a", "coverage_count_a"),
            ("member_b", "member_a", "stake_b", "win_points_b", "coverage_count_b"),
        ]:
            member_id = str(row.get(member_col, "")).strip().upper()
            clique = clique_lookup.get((int(row.get("draw_id")), member_id), {})
            position_count = int(row.get(coverage_col) or 0)
            stake = float(row.get(stake_col) or 0.0)
            win_points = float(row.get(win_col) or 0.0)
            projected.append(
                {
                    "member_id": member_id,
                    "ccs_id": row.get("ccs_a") if member_col == "member_a" else row.get("ccs_b"),
                    "draw_id": int(row.get("draw_id")),
                    "draw_date": row.get("draw_date"),
                    "best_partner_member_id": str(row.get(partner_col, "")).strip().upper(),
                    "position_count": position_count,
                    "stake": stake,
                    "win_points": win_points,
                    "net_result": win_points - stake,
                    "partition_score": min(position_count, 38 - position_count),
                    "best_partner_union_coverage": float(row.get("union_coverage_pct") or 0.0),
                    "best_partner_jaccard": float((row.get("overlap_count") or 0) / max(row.get("union_count") or 0, 1)),
                    "best_partner_per_position_ratio": float(row.get("ratio_similarity") or 0.0),
                    "best_partner_combined_bet_cv": 0.0,
                    "best_partner_pair_net": float(row.get("pair_net") or 0.0),
                    "best_partner_overlap_positions": int(row.get("overlap_count") or 0),
                    "best_partner_union_positions": int(row.get("union_count") or 0),
                    "is_exact_complementary_pair_in_draw": int(row.get("is_strict_match") or 0),
                    "is_near_complementary_pair_in_draw": int(row.get("is_nearmiss") or 0),
                    "n_low_overlap_partners_in_draw": 1,
                    "n_strict_pairs_in_draw": int(row.get("is_strict_collusion_pattern", row.get("is_strict_match")) or 0),
                    "clique_size_estimate": int(clique.get("clique_size_estimate", 1)),
                    "clique_total_stake_share": float(clique.get("clique_total_stake_share", 0.0)),
                    "clique_max_pair_rule_confidence": float(clique.get("clique_max_pair_rule_confidence", 0.0)),
                    "max_strict_pair_score_today": float(clique.get("max_strict_pair_score_today", 0.0)),
                    "_pair_risk_score": float(row.get("pair_risk_score") or 0.0),
                }
            )

    member_rows = pd.DataFrame(projected)
    if member_rows.empty:
        return ensure_stage1_schema(pd.DataFrame())
    member_rows = member_rows.sort_values(
        ["member_id", "draw_id", "_pair_risk_score", "best_partner_union_coverage"],
        ascending=[True, True, False, False],
    )
    counts = (
        member_rows.groupby(["member_id", "draw_id"], as_index=False)["n_low_overlap_partners_in_draw"]
        .sum()
        .rename(columns={"n_low_overlap_partners_in_draw": "_partner_count"})
    )
    best = member_rows.drop_duplicates(["member_id", "draw_id"], keep="first").merge(
        counts,
        on=["member_id", "draw_id"],
        how="left",
    )
    best["n_low_overlap_partners_in_draw"] = pd.to_numeric(best["_partner_count"], errors="coerce").fillna(1).astype(int)
    strict_counts = (
        member_rows.groupby(["member_id", "draw_id"], as_index=False)["n_strict_pairs_in_draw"]
        .sum()
        .rename(columns={"n_strict_pairs_in_draw": "_strict_pair_count"})
    )
    best = best.merge(strict_counts, on=["member_id", "draw_id"], how="left")
    best["n_strict_pairs_in_draw"] = pd.to_numeric(best["_strict_pair_count"], errors="coerce").fillna(0).astype(int)
    best = best.drop(columns=["_pair_risk_score", "_partner_count", "_strict_pair_count"], errors="ignore")
    out = ensure_stage1_schema(best)
    return add_rolling_context(out, pd.DataFrame()) if rolling_context else out


def project_clique_scores_to_member_draw_rows(
    clique_df: pd.DataFrame,
    candidate_df: pd.DataFrame | None = None,
    *,
    rolling_context: bool = True,
) -> pd.DataFrame:
    if clique_df.empty:
        return ensure_stage1_schema(pd.DataFrame())

    candidate_lookup = _candidate_member_lookup(candidate_df)
    projected: list[dict[str, Any]] = []
    for row in clique_df.to_dict("records"):
        members = [str(member).strip().upper() for member in row.get("member_ids", []) if str(member).strip()]
        if not members:
            continue
        draw_id = int(row.get("draw_id"))
        clique_size = int(row.get("clique_size") or len(members))
        group_stake = float(row.get("group_total_stake") or 0.0)
        score = float(row.get("clique_risk_score") or row.get("rule_confidence") or 0.0)
        for member_id in members:
            context = candidate_lookup.get((draw_id, member_id), {})
            partners = sorted(member for member in members if member != member_id)
            stake = float(context.get("stake", 0.0))
            win_points = float(context.get("win_points", 0.0))
            position_count = int(context.get("position_count", 0))
            projected.append(
                {
                    "member_id": member_id,
                    "ccs_id": context.get("ccs_id", row.get("ccs_id")),
                    "draw_id": draw_id,
                    "draw_date": row.get("draw_date"),
                    "best_partner_member_id": partners[0] if partners else None,
                    "position_count": position_count,
                    "stake": stake,
                    "win_points": win_points,
                    "net_result": win_points - stake,
                    "partition_score": min(position_count, 38 - position_count),
                    "best_partner_union_coverage": float(row.get("union_coverage_pct") or 0.0),
                    "best_partner_jaccard": float(
                        (row.get("duplicate_position_count") or 0) / max(row.get("union_count") or 0, 1)
                    ),
                    "best_partner_per_position_ratio": float(row.get("avg_amount_ratio") or 0.0),
                    "best_partner_combined_bet_cv": float(row.get("combined_bet_cv") or 0.0),
                    "best_partner_pair_net": float(row.get("group_net") or 0.0),
                    "best_partner_overlap_positions": int(row.get("duplicate_position_count") or 0),
                    "best_partner_union_positions": int(row.get("union_count") or 0),
                    "is_exact_complementary_pair_in_draw": int((row.get("union_count") or 0) >= 38),
                    "is_near_complementary_pair_in_draw": int((row.get("union_count") or 0) >= 36),
                    "n_low_overlap_partners_in_draw": max(clique_size - 1, 0),
                    "n_strict_pairs_in_draw": max(clique_size - 1, 0)
                    if int(row.get("is_strict_clique") or 0)
                    else 0,
                    "clique_size_estimate": clique_size,
                    "clique_total_stake_share": float(stake / group_stake) if group_stake > 0 else 0.0,
                    "clique_max_pair_rule_confidence": float(row.get("rule_confidence") or 0.0),
                    "max_strict_pair_score_today": score,
                    "stage1_score": score,
                    "_partner_count": max(clique_size - 1, 0),
                }
            )

    if not projected:
        return ensure_stage1_schema(pd.DataFrame())
    frame = pd.DataFrame(projected)
    frame["stage1_score"] = pd.to_numeric(frame["stage1_score"], errors="coerce").fillna(0.0)
    frame = frame.sort_values(
        ["member_id", "draw_id", "stage1_score", "best_partner_union_coverage"],
        ascending=[True, True, False, False],
    )
    partner_counts = (
        frame.groupby(["member_id", "draw_id"], as_index=False)["_partner_count"]
        .sum()
        .rename(columns={"_partner_count": "_clique_partner_count"})
    )
    best = frame.drop_duplicates(["member_id", "draw_id"], keep="first").merge(
        partner_counts,
        on=["member_id", "draw_id"],
        how="left",
    )
    best["n_low_overlap_partners_in_draw"] = pd.to_numeric(
        best["_clique_partner_count"], errors="coerce"
    ).fillna(best["n_low_overlap_partners_in_draw"]).astype(int)
    scores = best[["member_id", "draw_id", "stage1_score"]].copy()
    out = ensure_stage1_schema(best.drop(columns=["_partner_count", "_clique_partner_count"], errors="ignore"))
    out = out.merge(scores, on=["member_id", "draw_id"], how="left")
    out["stage1_score"] = pd.to_numeric(out["stage1_score"], errors="coerce").fillna(0.0)
    return add_rolling_context(out, pd.DataFrame()) if rolling_context else out


def compute_partnership_features_from_candidates(
    candidate_df: pd.DataFrame,
    *,
    rule_config: PairRuleConfig | dict[str, Any] | None = None,
    clique_rule_config: CliqueRuleConfig | dict[str, Any] | None = None,
    mode: str = "inference",
    ordinary_negative_sample: int = 0,
    rolling_context: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_df = compute_pair_rows_from_candidates(
        candidate_df,
        rule_config=rule_config,
        mode=mode,
        ordinary_negative_sample=ordinary_negative_sample,
        random_seed=random_seed,
    )
    clique_df = compute_clique_rows_from_candidates(candidate_df, rule_config=clique_rule_config)
    pair_stage1 = project_pair_scores_to_member_draw_rows(pair_df, rolling_context=False)
    clique_stage1 = project_clique_scores_to_member_draw_rows(
        clique_df,
        candidate_df,
        rolling_context=False,
    )
    stage1_member_df = _combine_stage1_signal_rows(pair_stage1, clique_stage1)
    if rolling_context:
        stage1_member_df = add_rolling_context(stage1_member_df, pd.DataFrame())
    return stage1_member_df, pair_df, pd.DataFrame()


def build_player_draw_vectors(raw_df: pd.DataFrame, *, timestamp_field: str = "trans_date") -> pd.DataFrame:
    from fraud_detection.components.feature_engineering import normalize_bet_position, parse_bets

    rows: list[dict[str, Any]] = []
    if raw_df.empty:
        return pd.DataFrame(columns=["member_id", "draw_id", "draw_date", "position_mask", "position_amounts", "position_count", "stake", "win_points", "net_result"])
    for row in raw_df.to_dict("records"):
        amounts = [0.0] * 38
        mask = 0
        for bet in parse_bets(row.get("bets")):
            try:
                amount = float(bet.get("bet_amount", 0) or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if amount <= 0:
                continue
            pos = normalize_bet_position(bet.get("number"))
            idx = _position_index(pos)
            if idx is None:
                continue
            amounts[idx] += amount
            mask |= 1 << idx
        stake = float(pd.to_numeric(pd.Series([row.get("total_bet_amount", sum(amounts))]), errors="coerce").fillna(sum(amounts)).iloc[0])
        win_points = float(pd.to_numeric(pd.Series([row.get("win_points", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        rows.append(
            {
                "member_id": str(row.get("member_id", "")).strip().upper(),
                "draw_id": int(pd.to_numeric(pd.Series([row.get("draw_id")]), errors="coerce").dropna().iloc[0]),
                "draw_date": pd.to_datetime(row.get(timestamp_field), errors="coerce", utc=True),
                "position_mask": int(mask),
                "position_amounts": amounts,
                "position_count": int(mask.bit_count()),
                "stake": stake,
                "win_points": win_points,
                "net_result": win_points - stake,
            }
        )
    return pd.DataFrame(rows)


def build_stage1_features(
    player_draw_vectors: pd.DataFrame,
    partnership_df: pd.DataFrame,
    *,
    min_position_count: int = 4,
    max_position_count: int = 34,
    max_candidates_per_draw: int = 80,
    max_jaccard: float = 0.10,
    expandable_pair_min_coverage: float = 0.50,
    section_b_min_stake: float = 1000.0,
    section_b_min_per_position_ratio: float = 0.80,
    section_b_max_combined_bet_cv: float = 0.35,
    section_b_min_pair_net: float = 0.0,
    include_rolling_context: bool = True,
    candidate_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = player_draw_vectors.copy()
    if base.empty:
        return _empty_stage1_frame(), _empty_pair_events()
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for draw_id, draw_rows in base.groupby("draw_id", sort=False):
        candidates = draw_rows.loc[
            (draw_rows["position_count"] >= min_position_count)
            & (draw_rows["position_count"] <= max_position_count)
            & (draw_rows["stake"] > 0)
        ].copy()
        candidate_member_ids = set(candidates["member_id"].astype(str))
        if not candidates.empty:
            candidates["partition_score"] = candidates["position_count"].map(lambda v: min(int(v), 38 - int(v)))
            candidates = candidates.sort_values(["partition_score", "position_count", "stake"], ascending=[False, False, False]).head(max_candidates_per_draw)
            candidate_member_ids = set(candidates["member_id"].astype(str))
        per_member = {
            str(row.member_id): _neutral_partner_metrics()
            for row in draw_rows.itertuples(index=False)
        }
        for left, right in combinations(candidates.to_dict("records"), 2):
            metric = _pair_metric(left, right)
            low_overlap = metric["jaccard"] <= max_jaccard and metric["union_coverage"] >= expandable_pair_min_coverage
            strict_complement = (
                metric["overlap_positions"] == 0
                and metric["union_coverage"] >= 0.95
                and metric["pair_stake"] >= section_b_min_stake
                and metric["per_position_ratio"] >= section_b_min_per_position_ratio
                and metric["combined_bet_cv"] <= section_b_max_combined_bet_cv
                and metric["pair_net"] >= section_b_min_pair_net
            )
            metric["is_exact"] = int(strict_complement and metric["union_positions"] == 38)
            metric["is_near"] = int(strict_complement and 36 <= metric["union_positions"] <= 37)
            if low_overlap:
                events.append(metric)
                per_member[str(left["member_id"])]["n_low_overlap_partners_in_draw"] += 1
                per_member[str(right["member_id"])]["n_low_overlap_partners_in_draw"] += 1
            for member, partner in [(str(left["member_id"]), str(right["member_id"])), (str(right["member_id"]), str(left["member_id"]))]:
                candidate = {
                    "best_partner_member_id": partner,
                    "best_partner_union_coverage": metric["union_coverage"],
                    "best_partner_jaccard": metric["jaccard"],
                    "best_partner_per_position_ratio": metric["per_position_ratio"],
                    "best_partner_combined_bet_cv": metric["combined_bet_cv"],
                    "best_partner_pair_net": metric["pair_net"],
                    "best_partner_overlap_positions": metric["overlap_positions"],
                    "best_partner_union_positions": metric["union_positions"],
                    "is_exact_complementary_pair_in_draw": metric["is_exact"],
                    "is_near_complementary_pair_in_draw": metric["is_near"],
                }
                if _partner_score(candidate) > _partner_score(per_member[member]):
                    per_member[member].update(candidate)
        for row in draw_rows.itertuples(index=False):
            if candidate_only and str(row.member_id) not in candidate_member_ids:
                continue
            rows.append(
                {
                    "member_id": str(row.member_id),
                    "draw_id": int(row.draw_id),
                    "draw_date": row.draw_date,
                    "position_count": int(row.position_count),
                    "stake": float(row.stake),
                    "win_points": float(row.win_points),
                    "net_result": float(row.net_result),
                    "partition_score": min(int(row.position_count), 38 - int(row.position_count)),
                    **per_member[str(row.member_id)],
                }
            )
    features = pd.DataFrame(rows)
    if include_rolling_context:
        features = add_rolling_context(features, partnership_df)
    return ensure_stage1_schema(features), pd.DataFrame(events) if events else _empty_pair_events()


def add_rolling_context(features: pd.DataFrame, partnership_df: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    out = features.sort_values(["member_id", "draw_date", "draw_id"]).copy()
    if "n_strict_pairs_in_draw" not in out.columns:
        out["n_strict_pairs_in_draw"] = 0
    if "n_low_overlap_partners_in_draw" not in out.columns:
        out["n_low_overlap_partners_in_draw"] = 0
    if "best_partner_member_id" not in out.columns:
        out["best_partner_member_id"] = None
    results = []
    for _, group in out.groupby("member_id", sort=False):
        group = group.copy()
        dates = pd.to_datetime(group["draw_date"], errors="coerce", utc=True)
        recurrence_1d = []
        recurrence_3d = []
        distinct_partners_1d = []
        distinct_partners_3d = []
        recurrences = []
        distinct_partners = []
        min_jaccards = []
        comp_pct = []
        for idx, current_date in enumerate(dates):
            if pd.isna(current_date):
                hist = group.iloc[0:0]
                hist_1d = hist
                hist_3d = hist
            else:
                hist = group.loc[(dates < current_date) & (dates >= current_date - pd.Timedelta(days=7))]
                hist_1d = group.loc[(dates < current_date) & (dates >= current_date - pd.Timedelta(days=1))]
                hist_3d = group.loc[(dates < current_date) & (dates >= current_date - pd.Timedelta(days=3))]
            recurrence_1d.append(int((pd.to_numeric(hist_1d.get("n_strict_pairs_in_draw", 0), errors="coerce").fillna(0) > 0).sum()))
            recurrence_3d.append(int((pd.to_numeric(hist_3d.get("n_strict_pairs_in_draw", 0), errors="coerce").fillna(0) > 0).sum()))
            distinct_partners_1d.append(int(hist_1d.get("best_partner_member_id", pd.Series(dtype=object)).dropna().astype(str).nunique()))
            distinct_partners_3d.append(int(hist_3d.get("best_partner_member_id", pd.Series(dtype=object)).dropna().astype(str).nunique()))
            recurrences.append(int((pd.to_numeric(hist.get("n_low_overlap_partners_in_draw", 0), errors="coerce").fillna(0) > 0).sum()))
            distinct_partners.append(int(hist.get("best_partner_member_id", pd.Series(dtype=object)).dropna().astype(str).nunique()))
            jacc = pd.to_numeric(hist.get("best_partner_jaccard", pd.Series(dtype=float)), errors="coerce")
            min_jaccards.append(float(jacc.min()) if jacc.notna().any() else 1.0)
            comp = pd.to_numeric(hist.get("is_exact_complementary_pair_in_draw", pd.Series(dtype=float)), errors="coerce").fillna(0)
            comp_pct.append(float(comp.mean()) if len(comp) else 0.0)
        group["recurrence_1d"] = recurrence_1d
        group["recurrence_3d"] = recurrence_3d
        group["distinct_partners_1d"] = distinct_partners_1d
        group["distinct_partners_3d"] = distinct_partners_3d
        group["rolling_partnership_recurrence_7d"] = recurrences
        group["rolling_distinct_partners_7d"] = distinct_partners
        group["rolling_min_jaccard_7d"] = min_jaccards
        group["rolling_pct_draws_with_complementary_pair_7d"] = comp_pct
        results.append(group)
    return pd.concat(results, ignore_index=True)


def build_stage2_training_frame(
    stage1_predictions: pd.DataFrame,
    *,
    partnership_features: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    gold_members: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate OOF Stage 1 predictions into member-level Stage 2 features."""
    if stage1_predictions.empty:
        out = pd.DataFrame(columns=["member_id", *STAGE2_FEATURE_COLUMNS, "label_gold_member"])
        return out

    preds = stage1_predictions.copy()
    preds["member_id"] = preds["member_id"].astype(str).str.strip().str.upper()
    preds["stage1_score"] = pd.to_numeric(preds.get("stage1_score", 0.0), errors="coerce").fillna(0.0)
    preds["draw_date"] = pd.to_datetime(preds.get("draw_date"), errors="coerce", utc=True)
    if "best_partner_member_id" not in preds.columns:
        preds["best_partner_member_id"] = None

    agg_spec = {
        "max_stage1_score": ("stage1_score", "max"),
        "mean_stage1_score": ("stage1_score", "mean"),
        "n_draws_stage1_above_0p5": ("stage1_score", lambda s: int((s >= 0.5).sum())),
        "n_draws_stage1_above_0p9": ("stage1_score", lambda s: int((s >= 0.9).sum())),
        "longest_consecutive_high_stage1_streak": ("stage1_score", lambda s: _longest_streak(s >= 0.5)),
    }
    for col in [
        "max_strict_pair_score_today",
        "n_strict_pairs_in_draw",
        "clique_size_estimate",
        "clique_total_stake_share",
        "clique_max_pair_rule_confidence",
        "recurrence_1d",
        "recurrence_3d",
        "distinct_partners_1d",
        "distinct_partners_3d",
        "ccs_profit_share_1d",
        "ccs_profit_share_7d",
        "ccs_total_profit_1d",
        "ccs_total_profit_7d",
        "ccs_member_count_1d",
        "ccs_member_count_7d",
        "ccs_high_concentration_1d",
        "ccs_high_concentration_7d",
        "ccs_solo_member_1d",
        "ccs_solo_member_7d",
    ]:
        if col in preds.columns:
            agg_spec[col] = (col, "max")

    agg = preds.sort_values(["member_id", "draw_date", "draw_id"]).groupby("member_id", as_index=False).agg(**agg_spec)
    high = preds.loc[preds["stage1_score"] >= 0.5]
    distinct = (
        high.loc[high["best_partner_member_id"].notna() & high["best_partner_member_id"].astype(str).ne("")]
        .groupby("member_id")["best_partner_member_id"]
        .nunique()
        .reset_index(name="n_distinct_high_score_partners")
    )
    agg = agg.merge(distinct, on="member_id", how="left")

    if partnership_features is not None and not partnership_features.empty:
        pf = partnership_features.copy()
        pf["member_id"] = pf["member_id"].astype(str).str.strip().str.upper()
        agg = agg.merge(pf, on="member_id", how="left")

    for col in STAGE2_FEATURE_COLUMNS:
        if col not in agg.columns:
            agg[col] = 0.0
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0)

    members = gold_members
    if members is None and labels is not None and not labels.empty:
        label_col = "member_id" if "member_id" in labels.columns else "MEMBER_ID"
        members = set(labels[label_col].astype(str).str.strip().str.upper())
    agg["label_gold_member"] = agg["member_id"].isin(members or set()).astype(int)
    return agg[["member_id", *STAGE2_FEATURE_COLUMNS, "label_gold_member"]]


def build_live_stage2_frame(
    current_draw_scores: pd.DataFrame,
    *,
    stage1_history: pd.DataFrame | None = None,
    partnership_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Stage 2 feature rows for live draw scoring.

    The current draw is combined with recent persisted Stage 1 history. Cold
    starts receive neutral history through normal aggregation.
    """
    frames = [current_draw_scores]
    if stage1_history is not None and not stage1_history.empty:
        frames.append(stage1_history)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return build_stage2_training_frame(
        combined,
        partnership_features=partnership_features,
        labels=None,
        gold_members=set(),
    ).drop(columns=["label_gold_member"], errors="ignore")


def ensure_stage1_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for key in ["member_id", "draw_id", "draw_date"]:
        if key not in out.columns:
            out[key] = pd.NA
    if "ccs_id" not in out.columns:
        out["ccs_id"] = None
    if "best_partner_member_id" not in out.columns:
        out["best_partner_member_id"] = None
    for col in STAGE1_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["member_id"] = out["member_id"].astype(str).str.strip().str.upper()
    return out[["member_id", "ccs_id", "draw_id", "draw_date", "best_partner_member_id", *STAGE1_FEATURE_COLUMNS]]


def _candidate_member_lookup(candidate_df: pd.DataFrame | None) -> dict[tuple[int, str], dict[str, Any]]:
    if candidate_df is None or candidate_df.empty:
        return {}
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for row in candidate_df.to_dict("records"):
        draw_id_value = pd.to_numeric(row.get("draw_id"), errors="coerce")
        if pd.isna(draw_id_value):
            continue
        draw_id = int(draw_id_value)
        members = [str(value).strip().upper() for value in row.get("member_ids", [])]
        ccs_ids = list(row.get("ccs_ids", []))
        stakes = list(row.get("total_bet_amounts", []))
        wins = list(row.get("win_points", []))
        coverage_bytes = list(row.get("coverage_bytes", []))
        for idx, member_id in enumerate(members):
            if not member_id:
                continue
            raw_coverage = coverage_bytes[idx] if idx < len(coverage_bytes) else b""
            coverage = np.frombuffer(bytes(raw_coverage), dtype=np.uint8) if raw_coverage is not None else np.array([])
            lookup[(draw_id, member_id)] = {
                "ccs_id": None if idx >= len(ccs_ids) else ccs_ids[idx],
                "stake": float(stakes[idx] or 0.0) if idx < len(stakes) else 0.0,
                "win_points": float(wins[idx] or 0.0) if idx < len(wins) else 0.0,
                "position_count": int((coverage > 0).sum()) if coverage.size else 0,
            }
    return lookup


def _combine_stage1_signal_rows(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return ensure_stage1_schema(pd.DataFrame())
    for frame in usable:
        if "stage1_score" not in frame.columns:
            frame["stage1_score"] = 0.0
        frame["stage1_score"] = pd.to_numeric(frame["stage1_score"], errors="coerce").fillna(0.0)
    combined = pd.concat(usable, ignore_index=True)
    combined["member_id"] = combined["member_id"].astype(str).str.strip().str.upper()
    combined["draw_id"] = pd.to_numeric(combined["draw_id"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["draw_id"]).copy()
    if combined.empty:
        return ensure_stage1_schema(pd.DataFrame())
    combined["draw_id"] = combined["draw_id"].astype(int)
    for column in ["n_low_overlap_partners_in_draw", "n_strict_pairs_in_draw"]:
        if column not in combined.columns:
            combined[column] = 0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0).astype(int)
    aggregate_counts = combined.groupby(["member_id", "draw_id"], as_index=False).agg(
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
        aggregate_counts,
        on=["member_id", "draw_id"],
        how="left",
    )
    best["n_low_overlap_partners_in_draw"] = best["_n_low_overlap_partners_in_draw"]
    best["n_strict_pairs_in_draw"] = best["_n_strict_pairs_in_draw"]
    best["clique_size_estimate"] = best["_clique_size_estimate"].fillna(best.get("clique_size_estimate", 1))
    best["clique_total_stake_share"] = best["_clique_total_stake_share"].fillna(
        best.get("clique_total_stake_share", 0.0)
    )
    best["clique_max_pair_rule_confidence"] = best["_clique_max_pair_rule_confidence"].fillna(
        best.get("clique_max_pair_rule_confidence", 0.0)
    )
    best["max_strict_pair_score_today"] = best["_max_strict_pair_score_today"].fillna(
        best.get("max_strict_pair_score_today", 0.0)
    )
    scores = best[["member_id", "draw_id", "_stage1_score"]].rename(columns={"_stage1_score": "stage1_score"})
    out = ensure_stage1_schema(best.drop(columns=[column for column in best.columns if column.startswith("_")], errors="ignore"))
    out = out.merge(scores, on=["member_id", "draw_id"], how="left")
    out["stage1_score"] = pd.to_numeric(out["stage1_score"], errors="coerce").fillna(0.0)
    return out


def save_partnership_feature_artifacts(
    output_dir: Path,
    *,
    stage1_features: pd.DataFrame,
    pair_events: pd.DataFrame,
    partnership_table: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage1_features": output_dir / "stage1_features.parquet",
        "pair_events": output_dir / "pair_events.parquet",
        "partnership_table": output_dir / "partnership_table.parquet",
    }
    stage1_features.to_parquet(paths["stage1_features"], index=False)
    pair_events.to_parquet(paths["pair_events"], index=False)
    partnership_table.to_parquet(paths["partnership_table"], index=False)
    return {key: str(path) for key, path in paths.items()}


def _coerce_thresholds(value: PartnershipThresholds | dict[str, Any] | None) -> PartnershipThresholds:
    if value is None:
        return PartnershipThresholds()
    if isinstance(value, PartnershipThresholds):
        return value
    allowed = PartnershipThresholds().__dict__.keys()
    return PartnershipThresholds(**{k: v for k, v in dict(value).items() if k in allowed})


def _coerce_partnership_df(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy() if value is not None else pd.DataFrame()


def _empty_stage1_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["member_id", "ccs_id", "draw_id", "draw_date", "best_partner_member_id", *STAGE1_FEATURE_COLUMNS])


def _empty_pair_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["draw_id", "draw_date", "member_a", "member_b", "stake_a", "stake_b", "win_points_a", "win_points_b", "pair_stake", "pair_win_points", "union_coverage", "jaccard", "per_position_ratio", "combined_bet_cv", "pair_net", "overlap_positions", "union_positions", "is_exact", "is_near"])


def _longest_streak(values: pd.Series) -> int:
    best = 0
    current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _position_index(position: str | None) -> int | None:
    if position is None:
        return None
    if position == "0":
        return 0
    if position == "00":
        return 37
    if str(position).isdigit():
        number = int(position)
        if 1 <= number <= 36:
            return number
    return None


def _neutral_partner_metrics() -> dict[str, Any]:
    return {
        "best_partner_member_id": None,
        "best_partner_union_coverage": 0.0,
        "best_partner_jaccard": 1.0,
        "best_partner_per_position_ratio": 0.0,
        "best_partner_combined_bet_cv": 0.0,
        "best_partner_pair_net": 0.0,
        "best_partner_overlap_positions": 0,
        "best_partner_union_positions": 0,
        "is_exact_complementary_pair_in_draw": 0,
        "is_near_complementary_pair_in_draw": 0,
        "n_low_overlap_partners_in_draw": 0,
        "n_strict_pairs_in_draw": 0,
        "clique_size_estimate": 1,
        "clique_total_stake_share": 0.0,
        "clique_max_pair_rule_confidence": 0.0,
        "max_strict_pair_score_today": 0.0,
        "recurrence_1d": 0,
        "recurrence_3d": 0,
        "distinct_partners_1d": 0,
        "distinct_partners_3d": 0,
    }


def _strict_clique_lookup(pair_frame: pd.DataFrame) -> dict[tuple[int, str], dict[str, float | int]]:
    if pair_frame.empty:
        return {}
    strict_col = "is_strict_collusion_pattern" if "is_strict_collusion_pattern" in pair_frame.columns else "is_strict_match"
    strict = pair_frame.loc[pd.to_numeric(pair_frame.get(strict_col), errors="coerce").fillna(0).astype(int) == 1].copy()
    if strict.empty:
        return {}
    strict["draw_id"] = pd.to_numeric(strict["draw_id"], errors="coerce").astype("Int64")
    strict = strict.dropna(subset=["draw_id"])
    for col in ["stake_a", "stake_b", "rule_confidence", "pair_risk_score"]:
        if col not in strict.columns:
            strict[col] = 0.0
        strict[col] = pd.to_numeric(strict[col], errors="coerce").fillna(0.0)

    lookup: dict[tuple[int, str], dict[str, float | int]] = {}
    for draw_id, draw_pairs in strict.groupby("draw_id", sort=False):
        members = pd.unique(
            pd.concat(
                [
                    draw_pairs["member_a"].astype(str).str.strip().str.upper(),
                    draw_pairs["member_b"].astype(str).str.strip().str.upper(),
                ],
                ignore_index=True,
            )
        )
        member_stake = {member: 0.0 for member in members}
        partner_counts = {member: 0 for member in members}
        max_conf = {member: 0.0 for member in members}
        max_score = {member: 0.0 for member in members}
        for row in draw_pairs.to_dict("records"):
            a = str(row.get("member_a", "")).strip().upper()
            b = str(row.get("member_b", "")).strip().upper()
            stake_a = float(row.get("stake_a") or 0.0)
            stake_b = float(row.get("stake_b") or 0.0)
            conf = float(row.get("rule_confidence") or 0.0)
            score = float(row.get("pair_risk_score") or conf)
            member_stake[a] = max(member_stake.get(a, 0.0), stake_a)
            member_stake[b] = max(member_stake.get(b, 0.0), stake_b)
            partner_counts[a] = partner_counts.get(a, 0) + 1
            partner_counts[b] = partner_counts.get(b, 0) + 1
            max_conf[a] = max(max_conf.get(a, 0.0), conf)
            max_conf[b] = max(max_conf.get(b, 0.0), conf)
            max_score[a] = max(max_score.get(a, 0.0), score)
            max_score[b] = max(max_score.get(b, 0.0), score)
        total_stake = sum(member_stake.values())
        clique_size = int(len([member for member, count in partner_counts.items() if count > 0]))
        for member, stake in member_stake.items():
            lookup[(int(draw_id), member)] = {
                "clique_size_estimate": clique_size,
                "clique_total_stake_share": float(stake / total_stake) if total_stake > 0 else 0.0,
                "clique_max_pair_rule_confidence": float(max_conf.get(member, 0.0)),
                "max_strict_pair_score_today": float(max_score.get(member, 0.0)),
            }
    return lookup


def _pair_metric(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_mask = int(left["position_mask"])
    right_mask = int(right["position_mask"])
    union_positions = int((left_mask | right_mask).bit_count())
    overlap_positions = int((left_mask & right_mask).bit_count())
    union_coverage = union_positions / 38.0
    denom = union_positions
    jaccard = overlap_positions / denom if denom else 0.0
    avg_left = float(left["stake"]) / max(int(left["position_count"]), 1)
    avg_right = float(right["stake"]) / max(int(right["position_count"]), 1)
    per_position_ratio = min(avg_left, avg_right) / max(avg_left, avg_right) if max(avg_left, avg_right) > 0 else 0.0
    combined = np.array(left["position_amounts"], dtype=float) + np.array(right["position_amounts"], dtype=float)
    positive = combined[combined > 0]
    combined_bet_cv = float(np.std(positive) / np.mean(positive)) if len(positive) and np.mean(positive) else 0.0
    pair_net = float(left["win_points"] + right["win_points"] - left["stake"] - right["stake"])
    return {
        "draw_id": int(left["draw_id"]),
        "draw_date": left["draw_date"],
        "member_a": str(left["member_id"]),
        "member_b": str(right["member_id"]),
        "stake_a": float(left["stake"]),
        "stake_b": float(right["stake"]),
        "win_points_a": float(left["win_points"]),
        "win_points_b": float(right["win_points"]),
        "pair_stake": float(left["stake"] + right["stake"]),
        "pair_win_points": float(left["win_points"] + right["win_points"]),
        "union_coverage": float(union_coverage),
        "jaccard": float(jaccard),
        "per_position_ratio": float(per_position_ratio),
        "combined_bet_cv": float(combined_bet_cv),
        "pair_net": pair_net,
        "overlap_positions": overlap_positions,
        "union_positions": union_positions,
        "is_exact": 0,
        "is_near": 0,
    }


def _partner_score(metrics: dict[str, Any]) -> float:
    return (
        float(metrics.get("best_partner_union_coverage", 0.0)) * 3.0
        - float(metrics.get("best_partner_jaccard", 1.0))
        + 0.5 * float(metrics.get("is_exact_complementary_pair_in_draw", 0.0))
        + 0.25 * float(metrics.get("is_near_complementary_pair_in_draw", 0.0))
        + min(max(float(metrics.get("best_partner_pair_net", 0.0)), 0.0) / 10000.0, 0.25)
    )
