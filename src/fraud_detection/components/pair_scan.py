from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairRuleConfig:
    board_size: int = 38
    strict_min_ratio_similarity: float = 0.80
    strict_min_stake_ratio: float = 0.90
    strict_max_overlap_count: int = 2
    strict_min_pair_net_per_stake: float = -0.10
    nearmiss_min_union: int = 36
    nearmiss_max_overlap: int = 2
    nearmiss_min_ratio_similarity: float = 0.70
    nearmiss_min_pair_net_per_stake: float = -0.10
    nearmiss_require_different_ccs: bool = True
    min_total_bet_amount: float = 1000.0
    min_pair_total_bet_amount: float = 0.0
    stage1_flag_threshold: float = 0.70
    strict_inference_filter: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PAIR_FEATURE_COLUMNS = [
    "coverage_count_a",
    "coverage_count_b",
    "coverage_count_diff",
    "overlap_count",
    "union_count",
    "union_coverage_pct",
    "ratio_similarity",
    "pair_net",
    "pair_net_per_stake",
    "stake_a",
    "stake_b",
    "stake_ratio",
    "win_points_a",
    "win_points_b",
    "pair_different_ccs",
    "bet_amount_ratio_within_10pct",
    "is_full_board_pair",
    "is_min_overlap_pair",
    "is_strict_collusion_pattern",
    "rule_confidence",
    "is_strict_match",
    "is_nearmiss",
]


def build_draw_matrices(
    coverage_bytes: list[bytes],
    amount_vector: list[list[float]],
    *,
    board_size: int = 38,
) -> tuple[np.ndarray, np.ndarray]:
    coverage = np.array(
        [np.frombuffer(bytes(item), dtype=np.uint8).astype(bool) for item in coverage_bytes],
        dtype=bool,
    )
    amounts = np.array(amount_vector, dtype=np.float64)
    if coverage.size == 0:
        coverage = np.zeros((0, board_size), dtype=bool)
    if amounts.size == 0:
        amounts = np.zeros((0, board_size), dtype=np.float64)
    if coverage.ndim != 2 or coverage.shape[1] != board_size:
        raise ValueError(f"coverage matrix must have shape (N, {board_size}); got {coverage.shape}")
    if amounts.shape != coverage.shape:
        raise ValueError(f"amount matrix shape {amounts.shape} does not match coverage shape {coverage.shape}")
    return coverage, amounts


def per_position_ratio_similarity(amounts: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    if amounts.shape != coverage.shape:
        raise ValueError("amounts and coverage must have the same shape.")
    covered_counts = coverage.sum(axis=1)
    positive_sums = amounts.sum(axis=1)
    avg_positive = np.divide(
        positive_sums,
        covered_counts,
        out=np.zeros_like(positive_sums, dtype=np.float64),
        where=covered_counts > 0,
    )
    left = avg_positive[:, None]
    right = avg_positive[None, :]
    return np.divide(np.minimum(left, right), np.maximum(left, right), out=np.zeros((len(amounts), len(amounts))), where=np.maximum(left, right) > 0)


def compute_pair_metrics(
    coverage: np.ndarray,
    amounts: np.ndarray,
    total_bet_amounts: np.ndarray,
    win_points: np.ndarray,
) -> dict[str, np.ndarray]:
    if coverage.shape != amounts.shape:
        raise ValueError("coverage and amounts matrices must have matching shapes.")
    if len(total_bet_amounts) != coverage.shape[0] or len(win_points) != coverage.shape[0]:
        raise ValueError("stake and win arrays must match number of coverage rows.")
    coverage_i = coverage.astype(np.int32)
    overlap_count = coverage_i @ coverage_i.T
    cov_count = coverage_i.sum(axis=1)
    union_count = cov_count[:, None] + cov_count[None, :] - overlap_count
    pair_net = (
        (win_points[:, None] + win_points[None, :])
        - (total_bet_amounts[:, None] + total_bet_amounts[None, :])
    )
    return {
        "overlap_count": overlap_count,
        "union_count": union_count,
        "coverage_count": cov_count,
        "pair_net": pair_net,
        "ratio_similarity": per_position_ratio_similarity(amounts, coverage),
    }


def emit_pair_rows(
    draw_row: dict[str, Any] | pd.Series,
    config: PairRuleConfig | None = None,
    *,
    mode: str = "inference",
    ordinary_negative_sample: int = 0,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    cfg = config or PairRuleConfig()
    row = draw_row.to_dict() if isinstance(draw_row, pd.Series) else dict(draw_row)
    member_ids = [str(value).strip().upper() for value in row.get("member_ids", [])]
    raw_ccs_ids = list(row.get("ccs_ids", []))
    ccs_ids = [
        None if value is None or str(value).strip() == "" else str(value).strip().upper()
        for value in raw_ccs_ids
    ]
    total_bets = np.array(row.get("total_bet_amounts", []), dtype=np.float64)
    wins = np.array(row.get("win_points", []), dtype=np.float64)
    coverage_bytes = list(row.get("coverage_bytes", []))
    amount_vector = list(row.get("amount_vector", []))
    coverage, amounts = build_draw_matrices(coverage_bytes, amount_vector, board_size=cfg.board_size)
    n_players = coverage.shape[0]
    if not (len(member_ids) == len(total_bets) == len(wins) == n_players):
        raise ValueError("candidate draw row has misaligned player arrays.")
    if len(ccs_ids) < n_players:
        ccs_ids.extend([None] * (n_players - len(ccs_ids)))
    elif len(ccs_ids) > n_players:
        ccs_ids = ccs_ids[:n_players]
    if n_players < 2:
        return []

    metrics = compute_pair_metrics(coverage, amounts, total_bets, wins)
    upper_i, upper_j = np.triu_indices(n_players, k=1)
    both_staked = (total_bets[upper_i] >= cfg.min_total_bet_amount) & (total_bets[upper_j] >= cfg.min_total_bet_amount)
    union_count = metrics["union_count"][upper_i, upper_j]
    overlap_count = metrics["overlap_count"][upper_i, upper_j]
    ratio = metrics["ratio_similarity"][upper_i, upper_j]
    pair_net = metrics["pair_net"][upper_i, upper_j]
    pair_stake = total_bets[upper_i] + total_bets[upper_j]
    stake_ratio = np.divide(
        np.minimum(total_bets[upper_i], total_bets[upper_j]),
        np.maximum(total_bets[upper_i], total_bets[upper_j]),
        out=np.zeros_like(pair_stake, dtype=np.float64),
        where=np.maximum(total_bets[upper_i], total_bets[upper_j]) > 0,
    )
    pair_total_ok = pair_stake >= float(cfg.min_pair_total_bet_amount)
    pair_net_per_stake = np.divide(
        pair_net,
        pair_stake,
        out=np.zeros_like(pair_net, dtype=np.float64),
        where=pair_stake > 0,
    )
    ccs_i = np.array([ccs_ids[int(idx)] for idx in upper_i], dtype=object)
    ccs_j = np.array([ccs_ids[int(idx)] for idx in upper_j], dtype=object)
    pair_different_ccs = (ccs_i != None) & (ccs_j != None) & (ccs_i != ccs_j)  # noqa: E711
    is_full_board_pair = union_count == cfg.board_size
    is_min_overlap_pair = overlap_count <= cfg.strict_max_overlap_count
    bet_amount_ratio_within_10pct = stake_ratio >= cfg.strict_min_stake_ratio
    ratio_ok = ratio >= cfg.strict_min_ratio_similarity
    rule_confidence = (
        0.25 * is_full_board_pair.astype(float)
        + 0.25 * is_min_overlap_pair.astype(float)
        + 0.25 * bet_amount_ratio_within_10pct.astype(float)
        + 0.25 * np.clip(ratio / max(cfg.strict_min_ratio_similarity, 1e-9), 0.0, 1.0)
    )
    strict = (
        both_staked
        & pair_total_ok
        & is_full_board_pair
        & is_min_overlap_pair
        & bet_amount_ratio_within_10pct
        & ratio_ok
        & (pair_net_per_stake >= cfg.strict_min_pair_net_per_stake)
    )
    nearmiss = (
        both_staked
        & ~strict
        & (union_count >= cfg.nearmiss_min_union)
        & (overlap_count <= cfg.nearmiss_max_overlap)
        & bet_amount_ratio_within_10pct
        & (ratio >= cfg.nearmiss_min_ratio_similarity)
        & (pair_net_per_stake >= cfg.nearmiss_min_pair_net_per_stake)
    )
    if cfg.nearmiss_require_different_ccs:
        nearmiss = nearmiss & pair_different_ccs
    emit_mask = strict if (mode == "inference" and cfg.strict_inference_filter) else (strict | nearmiss)
    sampled_negative = np.zeros_like(emit_mask, dtype=bool)
    if mode == "training" and ordinary_negative_sample > 0:
        ordinary_idx = np.flatnonzero(both_staked & ~strict & ~nearmiss)
        if len(ordinary_idx):
            rng = np.random.default_rng(random_seed)
            chosen = rng.choice(ordinary_idx, size=min(int(ordinary_negative_sample), len(ordinary_idx)), replace=False)
            sampled_negative[chosen] = True
            emit_mask = emit_mask | sampled_negative

    draw_id = int(row["draw_id"])
    draw_date = pd.to_datetime(row.get("trans_date_min"), errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(emit_mask):
        i = int(upper_i[idx])
        j = int(upper_j[idx])
        row_pair_stake = float(total_bets[i] + total_bets[j])
        rows.append(
            {
                "draw_id": draw_id,
                "draw_date": draw_date,
                "member_a": member_ids[i],
                "member_b": member_ids[j],
                "ccs_a": ccs_ids[i],
                "ccs_b": ccs_ids[j],
                "coverage_count_a": int(metrics["coverage_count"][i]),
                "coverage_count_b": int(metrics["coverage_count"][j]),
                "coverage_count_diff": int(abs(metrics["coverage_count"][i] - metrics["coverage_count"][j])),
                "overlap_count": int(overlap_count[idx]),
                "union_count": int(union_count[idx]),
                "union_coverage_pct": float(union_count[idx] / cfg.board_size),
                "ratio_similarity": float(ratio[idx]),
                "pair_net": float(pair_net[idx]),
                "pair_net_per_stake": float(pair_net_per_stake[idx]),
                "stake_a": float(total_bets[i]),
                "stake_b": float(total_bets[j]),
                "stake_ratio": float(stake_ratio[idx]),
                "win_points_a": float(wins[i]),
                "win_points_b": float(wins[j]),
                "pair_different_ccs": int(pair_different_ccs[idx]),
                "bet_amount_ratio_within_10pct": int(bet_amount_ratio_within_10pct[idx]),
                "is_full_board_pair": int(is_full_board_pair[idx]),
                "is_min_overlap_pair": int(is_min_overlap_pair[idx]),
                "is_strict_collusion_pattern": int(strict[idx]),
                "rule_confidence": float(rule_confidence[idx]),
                "is_strict_match": int(strict[idx]),
                "is_nearmiss": int(nearmiss[idx]),
                "sampled_negative": int(sampled_negative[idx]),
                "pair_risk_score": 1.0 if strict[idx] else 0.0,
            }
        )
    return rows


def pair_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "draw_id",
        "draw_date",
        "member_a",
        "member_b",
        "ccs_a",
        "ccs_b",
        *PAIR_FEATURE_COLUMNS,
        "sampled_negative",
        "pair_risk_score",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def coerce_pair_rule_config(value: PairRuleConfig | dict[str, Any] | None) -> PairRuleConfig:
    if value is None:
        return PairRuleConfig()
    if isinstance(value, PairRuleConfig):
        return value
    allowed = PairRuleConfig.__dataclass_fields__
    return PairRuleConfig(**{key: val for key, val in dict(value).items() if key in allowed})
