from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.components.feature_engineering import normalize_bet_position, parse_bets

BOARD_SIZE = 38


def compute_draw_metrics_table(draw_rows: pd.DataFrame) -> pd.DataFrame:
    if draw_rows.empty:
        return pd.DataFrame(columns=_metric_columns())
    rows = []
    frame = draw_rows.copy()
    frame["draw_id"] = pd.to_numeric(frame.get("draw_id"), errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["draw_id"])
    for draw_id, group in frame.groupby("draw_id", sort=False):
        metrics = compute_draw_cohort_metrics(group)
        metrics["draw_id"] = int(draw_id)
        rows.append(metrics)
    return pd.DataFrame(rows, columns=_metric_columns()) if rows else pd.DataFrame(columns=_metric_columns())


def compute_draw_cohort_metrics(draw_rows: pd.DataFrame) -> dict[str, Any]:
    vectors = [_row_vector(row) for row in draw_rows.to_dict("records")]
    vectors = [item for item in vectors if item["member_id"]]
    if not vectors:
        return _empty_metrics()
    masks = [int(item["mask"]) for item in vectors]
    stakes = np.array([float(item["stake"]) for item in vectors], dtype=float)
    union_mask = 0
    for mask in masks:
        union_mask |= mask
    jaccards = []
    overlaps = []
    for left, right in combinations(masks, 2):
        union = int((left | right).bit_count())
        overlap = int((left & right).bit_count())
        overlaps.append(overlap)
        jaccards.append(overlap / union if union else 0.0)
    return {
        "cohort_size": int(len(vectors)),
        "union_position_count": int(union_mask.bit_count()),
        "union_position_coverage": float(union_mask.bit_count() / BOARD_SIZE),
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "min_pairwise_jaccard": float(np.min(jaccards)) if jaccards else 0.0,
        "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "total_cohort_stake": float(stakes.sum()),
        "mean_member_stake": float(stakes.mean()) if len(stakes) else 0.0,
        "stake_cv": float(stakes.std() / stakes.mean()) if len(stakes) and stakes.mean() else 0.0,
    }


def decision_verdict(fraud_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> dict[str, Any]:
    fraud = fraud_metrics.loc[pd.to_numeric(fraud_metrics.get("cohort_size"), errors="coerce").fillna(0).ge(2)]
    if fraud.empty:
        return {"verdict": "insufficient_evidence", "reason": "no labelled draws contain two or more fraud members"}
    fraud_coverage = pd.to_numeric(fraud.get("union_position_coverage"), errors="coerce").dropna()
    baseline_coverage = pd.to_numeric(baseline_metrics.get("union_position_coverage"), errors="coerce").dropna()
    fraud_median = float(fraud_coverage.median()) if not fraud_coverage.empty else 0.0
    baseline_median = float(baseline_coverage.median()) if not baseline_coverage.empty else 0.0
    if baseline_coverage.empty:
        return {
            "verdict": "needs_baseline",
            "reason": f"fraud median board coverage is {fraud_median:.3f}, but no baseline rows were available",
            "fraud_median_union_position_coverage": fraud_median,
        }
    lift = fraud_median - baseline_median
    verdict = "supports_collusion_rule" if fraud_median >= 0.90 and lift >= 0.10 else "weak_or_mixed_signal"
    return {
        "verdict": verdict,
        "reason": f"fraud median coverage {fraud_median:.3f}; baseline median {baseline_median:.3f}; delta {lift:.3f}",
        "fraud_median_union_position_coverage": fraud_median,
        "baseline_median_union_position_coverage": baseline_median,
        "coverage_delta": float(lift),
    }


def _row_vector(row: dict[str, Any]) -> dict[str, Any]:
    mask = 0
    for bet in parse_bets(row.get("bets")):
        pos = normalize_bet_position(bet.get("number"))
        idx = _position_index(pos)
        if idx is not None:
            mask |= 1 << idx
    return {
        "member_id": str(row.get("member_id", "")).strip().upper(),
        "mask": mask,
        "stake": pd.to_numeric(pd.Series([row.get("total_bet_amount", 0.0)]), errors="coerce").fillna(0.0).iloc[0],
    }


def _position_index(position: str | None) -> int | None:
    if position == "0":
        return 0
    if position == "00":
        return 37
    if position is not None and str(position).isdigit():
        number = int(position)
        if 1 <= number <= 36:
            return number
    return None


def _empty_metrics() -> dict[str, Any]:
    return {column: 0 for column in _metric_columns() if column != "draw_id"}


def _metric_columns() -> list[str]:
    return [
        "draw_id",
        "cohort_size",
        "union_position_count",
        "union_position_coverage",
        "mean_pairwise_jaccard",
        "min_pairwise_jaccard",
        "mean_pairwise_overlap",
        "total_cohort_stake",
        "mean_member_stake",
        "stake_cv",
    ]
