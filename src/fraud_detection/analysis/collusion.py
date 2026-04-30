from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.components.feature_engineering import parse_bets

ROULETTE_POSITION_COUNT = 38
ROULETTE_POSITIONS = {"0", "00", *{str(i) for i in range(1, 37)}}


def normalize_position(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"", "NAN", "NONE"}:
        return None
    if raw == "00":
        return "00"
    if raw in {"DOUBLE_ZERO", "DOUBLEZERO"}:
        return "00"
    if raw == "000":
        return "00"
    if raw.isdigit():
        try:
            number = int(raw)
        except ValueError:
            return raw
        if number == 0:
            return "0"
        if 1 <= number <= 36:
            return str(number)
    return raw


def bet_positions(bets: Any) -> set[str]:
    positions: set[str] = set()
    for bet in parse_bets(bets):
        try:
            amount = float(bet.get("bet_amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        if amount <= 0:
            continue
        position = normalize_position(bet.get("number"))
        if position is not None:
            positions.add(position)
    return positions


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mean_pairwise_jaccard(position_sets: list[set[str]]) -> float:
    pairs = list(combinations(position_sets, 2))
    if not pairs:
        return 0.0
    return float(np.mean([jaccard(a, b) for a, b in pairs]))


def compute_draw_cohort_metrics(draw_rows: pd.DataFrame) -> dict[str, float | int]:
    if draw_rows.empty:
        return {
            "cohort_size": 0,
            "union_position_count": 0,
            "union_position_coverage": 0.0,
            "mean_pairwise_jaccard": 0.0,
            "total_cohort_stake": 0.0,
            "mean_member_stake": 0.0,
        }

    working = draw_rows.copy()
    if "position_set" not in working.columns:
        working["position_set"] = working["bets"].apply(bet_positions)
    if "total_bet_amount" in working.columns:
        stakes = pd.to_numeric(working["total_bet_amount"], errors="coerce").fillna(0.0)
    else:
        stakes = pd.Series(0.0, index=working.index)

    position_sets = list(working["position_set"])
    union_positions = set().union(*position_sets) if position_sets else set()
    cohort_size = int(working["member_id"].astype(str).str.strip().str.upper().nunique())
    total_stake = float(stakes.sum())

    return {
        "cohort_size": cohort_size,
        "union_position_count": int(len(union_positions)),
        "union_position_coverage": float(len(union_positions) / ROULETTE_POSITION_COUNT),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(position_sets),
        "total_cohort_stake": total_stake,
        "mean_member_stake": float(total_stake / cohort_size) if cohort_size else 0.0,
    }


def compute_draw_metrics_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "draw_id",
                "cohort_size",
                "union_position_count",
                "union_position_coverage",
                "mean_pairwise_jaccard",
                "total_cohort_stake",
                "mean_member_stake",
            ]
        )

    working = rows.copy()
    working["member_id"] = working["member_id"].astype(str).str.strip().str.upper()
    working["draw_id"] = pd.to_numeric(working["draw_id"], errors="coerce").astype("Int64")
    working["position_set"] = working["bets"].apply(bet_positions)

    records = []
    for draw_id, draw_rows in working.groupby("draw_id", dropna=True):
        metrics = compute_draw_cohort_metrics(draw_rows)
        metrics["draw_id"] = int(draw_id)
        records.append(metrics)
    return pd.DataFrame(records)


def decision_verdict(fraud_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> dict[str, Any]:
    if fraud_metrics.empty:
        return {
            "verdict": "PIVOT",
            "reason": "No fraud cohort draws with 2+ labelled members were found.",
            "draws_with_2plus_fraud_cohort_pct": 0.0,
            "median_union_coverage_ratio": 0.0,
            "median_jaccard_ratio": 0.0,
        }

    fraud_2plus = fraud_metrics.loc[fraud_metrics["cohort_size"] >= 2]
    pct_2plus = float(len(fraud_2plus) / max(len(fraud_metrics), 1))
    fraud_cov = float(fraud_2plus["union_position_coverage"].median()) if not fraud_2plus.empty else 0.0
    fraud_jacc = float(fraud_2plus["mean_pairwise_jaccard"].median()) if not fraud_2plus.empty else 0.0
    baseline_cov = float(baseline_metrics["union_position_coverage"].median()) if not baseline_metrics.empty else 0.0
    baseline_jacc = float(baseline_metrics["mean_pairwise_jaccard"].median()) if not baseline_metrics.empty else 0.0
    cov_ratio = fraud_cov / baseline_cov if baseline_cov > 0 else 0.0
    jacc_ratio = fraud_jacc / baseline_jacc if baseline_jacc > 0 else 0.0

    # In roulette, random same-draw cohorts can also have high coverage when
    # draws are busy. Accept either relative coverage lift OR near-full fraud
    # coverage, but always require low overlap.
    coverage_signal = cov_ratio >= 1.5 or fraud_cov >= 0.90
    proceed = pct_2plus >= 0.10 and coverage_signal and jacc_ratio <= 0.7
    return {
        "verdict": "PROCEED" if proceed else "PIVOT",
        "reason": (
            "Fraud cohort draws show enough high-coverage, low-overlap collusion signal."
            if proceed
            else "Fraud cohort draws do not clear the configured collusion-signal thresholds."
        ),
        "draws_with_2plus_fraud_cohort_pct": pct_2plus,
        "median_union_coverage_fraud": fraud_cov,
        "median_union_coverage_baseline": baseline_cov,
        "median_union_coverage_ratio": cov_ratio,
        "median_jaccard_fraud": fraud_jacc,
        "median_jaccard_baseline": baseline_jacc,
        "median_jaccard_ratio": jacc_ratio,
    }
