from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.components.pair_scan import build_draw_matrices


@dataclass(frozen=True)
class CliqueRuleConfig:
    board_size: int = 38
    min_group_size: int = 3
    max_group_size: int = 4
    min_union_count: int = 34
    strict_union_count: int = 36
    max_duplicate_position_count: int = 2
    min_member_coverage_count: int = 2
    max_member_coverage_count: int = 24
    min_member_stake: float = 1000.0
    min_group_total_stake: float = 30_000.0
    min_avg_amount_ratio: float = 0.80
    min_total_stake_ratio: float = 0.90
    require_same_ccs: bool = True
    max_candidates_per_ccs: int = 16
    stage1_flag_threshold: float = 0.70
    section_team_enabled: bool = True
    min_section_team_clusters: int = 3
    min_section_team_distinct_ccs: int = 3
    min_section_team_union_count: int = 36
    min_duplicate_cluster_members: int = 3
    max_section_team_clusters: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLIQUE_EVENT_COLUMNS = [
    "draw_id",
    "draw_date",
    "member_ids",
    "ccs_id",
    "clique_size",
    "union_count",
    "union_coverage_pct",
    "duplicate_position_count",
    "max_position_overlap",
    "min_member_coverage_count",
    "max_member_coverage_count",
    "group_total_stake",
    "group_total_win",
    "group_net",
    "group_net_per_stake",
    "avg_amount_ratio",
    "total_stake_ratio",
    "combined_bet_cv",
    "rule_confidence",
    "clique_risk_score",
    "is_strict_clique",
    "is_nearmiss_clique",
    "cluster_count",
    "pattern_type",
]


def coerce_clique_rule_config(value: CliqueRuleConfig | dict[str, Any] | None) -> CliqueRuleConfig:
    if value is None:
        return CliqueRuleConfig()
    if isinstance(value, CliqueRuleConfig):
        return value
    allowed = CliqueRuleConfig.__dataclass_fields__
    return CliqueRuleConfig(**{key: val for key, val in dict(value).items() if key in allowed})


def clique_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in CLIQUE_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CLIQUE_EVENT_COLUMNS]


def emit_clique_rows(
    draw_row: dict[str, Any] | pd.Series,
    config: CliqueRuleConfig | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = coerce_clique_rule_config(config)
    row = draw_row.to_dict() if isinstance(draw_row, pd.Series) else dict(draw_row)
    member_ids = [str(value).strip().upper() for value in row.get("member_ids", [])]
    raw_ccs_ids = list(row.get("ccs_ids", []))
    ccs_ids = [
        None if value is None or str(value).strip() == "" else str(value).strip().upper()
        for value in raw_ccs_ids
    ]
    total_bets = np.array(row.get("total_bet_amounts", []), dtype=np.float64)
    wins = np.array(row.get("win_points", []), dtype=np.float64)
    coverage, amounts = build_draw_matrices(
        list(row.get("coverage_bytes", [])),
        list(row.get("amount_vector", [])),
        board_size=cfg.board_size,
    )
    n_players = coverage.shape[0]
    if not (len(member_ids) == len(total_bets) == len(wins) == n_players):
        raise ValueError("candidate draw row has misaligned player arrays.")
    if len(ccs_ids) < n_players:
        ccs_ids.extend([None] * (n_players - len(ccs_ids)))
    elif len(ccs_ids) > n_players:
        ccs_ids = ccs_ids[:n_players]
    if n_players < cfg.min_group_size:
        return []

    coverage_counts = coverage.sum(axis=1)
    eligible = [
        idx
        for idx in range(n_players)
        if total_bets[idx] >= cfg.min_member_stake
        and cfg.min_member_coverage_count <= coverage_counts[idx] <= cfg.max_member_coverage_count
    ]
    if len(eligible) < cfg.min_group_size:
        return []

    groups: dict[str, list[int]] = {}
    if cfg.require_same_ccs:
        for idx in eligible:
            ccs = ccs_ids[idx]
            if ccs:
                groups.setdefault(str(ccs), []).append(idx)
    else:
        groups["ALL"] = eligible

    draw_id = int(row["draw_id"])
    draw_date = pd.to_datetime(row.get("trans_date_min"), errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for ccs_id, indexes in groups.items():
        if len(indexes) < cfg.min_group_size:
            continue
        indexes = sorted(indexes, key=lambda idx: (total_bets[idx], coverage_counts[idx]), reverse=True)[
            : max(cfg.max_candidates_per_ccs, cfg.min_group_size)
        ]
        for group_size in range(cfg.min_group_size, min(cfg.max_group_size, len(indexes)) + 1):
            for combo in combinations(indexes, group_size):
                key = tuple(sorted(member_ids[idx] for idx in combo))
                if key in seen:
                    continue
                metrics = _clique_metrics(combo, coverage, amounts, total_bets, wins, cfg)
                strict = (
                    metrics["union_count"] >= cfg.strict_union_count
                    and metrics["duplicate_position_count"] <= cfg.max_duplicate_position_count
                    and metrics["avg_amount_ratio"] >= cfg.min_avg_amount_ratio
                    and metrics["total_stake_ratio"] >= cfg.min_total_stake_ratio
                    and metrics["group_total_stake"] >= cfg.min_group_total_stake
                )
                nearmiss = (
                    not strict
                    and metrics["union_count"] >= cfg.min_union_count
                    and metrics["duplicate_position_count"] <= cfg.max_duplicate_position_count
                    and metrics["avg_amount_ratio"] >= cfg.min_avg_amount_ratio
                    and metrics["total_stake_ratio"] >= cfg.min_total_stake_ratio
                    and metrics["group_total_stake"] >= cfg.min_group_total_stake
                )
                if not strict and not nearmiss:
                    continue
                seen.add(key)
                confidence = _rule_confidence(metrics, cfg)
                rows.append(
                    {
                        "draw_id": draw_id,
                        "draw_date": draw_date,
                        "member_ids": list(key),
                        "ccs_id": ccs_id if cfg.require_same_ccs else None,
                        **metrics,
                        "rule_confidence": float(confidence),
                        "clique_risk_score": 1.0 if strict else max(float(confidence), cfg.stage1_flag_threshold),
                        "is_strict_clique": int(strict),
                        "is_nearmiss_clique": int(nearmiss),
                    }
                )
    if cfg.section_team_enabled:
        rows.extend(
            _emit_section_team_rows(
                draw_id=draw_id,
                draw_date=draw_date,
                member_ids=member_ids,
                ccs_ids=ccs_ids,
                coverage=coverage,
                amounts=amounts,
                total_bets=total_bets,
                wins=wins,
                coverage_counts=coverage_counts,
                eligible=eligible,
                cfg=cfg,
                seen=seen,
            )
        )
    return rows


def _emit_section_team_rows(
    *,
    draw_id: int,
    draw_date: pd.Timestamp,
    member_ids: list[str],
    ccs_ids: list[str | None],
    coverage: np.ndarray,
    amounts: np.ndarray,
    total_bets: np.ndarray,
    wins: np.ndarray,
    coverage_counts: np.ndarray,
    eligible: list[int],
    cfg: CliqueRuleConfig,
    seen: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    signature_groups: dict[tuple[str, bytes], list[int]] = {}
    for idx in eligible:
        ccs = ccs_ids[idx]
        if not ccs:
            continue
        signature = coverage[idx].astype(np.uint8).tobytes()
        signature_groups.setdefault((str(ccs), signature), []).append(idx)

    clusters: list[dict[str, Any]] = []
    for (ccs_id, _signature), indexes in signature_groups.items():
        if len(indexes) < cfg.min_duplicate_cluster_members:
            continue
        indexes = sorted(indexes, key=lambda item: (total_bets[item], member_ids[item]), reverse=True)
        cluster_stake = float(total_bets[indexes].sum())
        if cluster_stake < cfg.min_group_total_stake:
            continue
        amount_ratio = _member_amount_ratio(indexes, coverage, amounts)
        if amount_ratio < cfg.min_avg_amount_ratio:
            continue
        total_stake_ratio = _member_total_stake_ratio(indexes, total_bets)
        if total_stake_ratio < cfg.min_total_stake_ratio:
            continue
        mask = coverage[indexes[0]].copy()
        clusters.append(
            {
                "ccs_id": ccs_id,
                "indexes": indexes,
                "coverage_mask": mask,
                "coverage_count": int(mask.sum()),
                "cluster_stake": cluster_stake,
                "total_stake_ratio": total_stake_ratio,
            }
        )

    if len(clusters) < cfg.min_section_team_clusters:
        return []

    selected: list[dict[str, Any]] = []
    union = np.zeros(cfg.board_size, dtype=bool)
    for cluster in sorted(clusters, key=lambda item: (item["coverage_count"], item["cluster_stake"]), reverse=True):
        adds_positions = int((cluster["coverage_mask"] & ~union).sum())
        if adds_positions <= 0 and len(selected) >= cfg.min_section_team_clusters:
            continue
        selected.append(cluster)
        union |= cluster["coverage_mask"]
        if len(selected) >= cfg.max_section_team_clusters and int(union.sum()) >= cfg.min_section_team_union_count:
            break

    if len(selected) < cfg.min_section_team_clusters:
        return []
    if len({str(cluster["ccs_id"]) for cluster in selected}) < cfg.min_section_team_distinct_ccs:
        return []
    union_count = int(union.sum())
    if union_count < cfg.min_section_team_union_count:
        return []

    indexes = tuple(idx for cluster in selected for idx in cluster["indexes"])
    key = tuple(sorted(member_ids[idx] for idx in indexes))
    if key in seen:
        return []
    metrics = _clique_metrics(indexes, coverage, amounts, total_bets, wins, cfg)
    metrics["total_stake_ratio"] = float(
        min(float(cluster["total_stake_ratio"]) for cluster in selected)
    )
    strict = (
        metrics["union_count"] >= cfg.strict_union_count
        and metrics["avg_amount_ratio"] >= cfg.min_avg_amount_ratio
        and metrics["group_total_stake"] >= cfg.min_group_total_stake
    )
    nearmiss = (
        not strict
        and metrics["union_count"] >= cfg.min_union_count
        and metrics["avg_amount_ratio"] >= cfg.min_avg_amount_ratio
        and metrics["group_total_stake"] >= cfg.min_group_total_stake
    )
    if not strict and not nearmiss:
        return []

    seen.add(key)
    confidence = _rule_confidence(metrics, cfg)
    return [
        {
            "draw_id": draw_id,
            "draw_date": draw_date,
            "member_ids": list(key),
            "ccs_id": "MULTI_CCS",
            **metrics,
            "rule_confidence": float(confidence),
            "clique_risk_score": 1.0 if strict else max(float(confidence), cfg.stage1_flag_threshold),
            "is_strict_clique": int(strict),
            "is_nearmiss_clique": int(nearmiss),
            "cluster_count": int(len(selected)),
            "pattern_type": "section_team",
        }
    ]


def _clique_metrics(
    indexes: tuple[int, ...],
    coverage: np.ndarray,
    amounts: np.ndarray,
    total_bets: np.ndarray,
    wins: np.ndarray,
    cfg: CliqueRuleConfig,
) -> dict[str, float | int]:
    group_coverage = coverage[list(indexes)]
    coverage_sum = group_coverage.sum(axis=0)
    union_mask = coverage_sum > 0
    union_count = int(union_mask.sum())
    duplicate_count = int((coverage_sum > 1).sum())
    positive_amounts = [
        float(amounts[idx][coverage[idx]].mean()) if coverage[idx].any() else 0.0
        for idx in indexes
    ]
    positive_amounts = [value for value in positive_amounts if value > 0]
    avg_amount_ratio = (
        min(positive_amounts) / max(positive_amounts)
        if positive_amounts and max(positive_amounts) > 0
        else 0.0
    )
    combined = amounts[list(indexes)].sum(axis=0)
    positive_combined = combined[combined > 0]
    combined_cv = (
        float(np.std(positive_combined) / np.mean(positive_combined))
        if len(positive_combined) and np.mean(positive_combined) > 0
        else 0.0
    )
    total_stake = float(total_bets[list(indexes)].sum())
    total_win = float(wins[list(indexes)].sum())
    total_stake_ratio = _member_total_stake_ratio(indexes, total_bets)
    return {
        "clique_size": int(len(indexes)),
        "union_count": union_count,
        "union_coverage_pct": float(union_count / cfg.board_size),
        "duplicate_position_count": duplicate_count,
        "max_position_overlap": int(coverage_sum.max()) if len(coverage_sum) else 0,
        "min_member_coverage_count": int(group_coverage.sum(axis=1).min()) if len(indexes) else 0,
        "max_member_coverage_count": int(group_coverage.sum(axis=1).max()) if len(indexes) else 0,
        "group_total_stake": total_stake,
        "group_total_win": total_win,
        "group_net": float(total_win - total_stake),
        "group_net_per_stake": float((total_win - total_stake) / total_stake) if total_stake > 0 else 0.0,
        "avg_amount_ratio": float(avg_amount_ratio),
        "total_stake_ratio": float(total_stake_ratio),
        "combined_bet_cv": float(combined_cv),
    }


def _member_amount_ratio(
    indexes: list[int] | tuple[int, ...],
    coverage: np.ndarray,
    amounts: np.ndarray,
) -> float:
    positive_amounts = [
        float(amounts[idx][coverage[idx]].mean()) if coverage[idx].any() else 0.0
        for idx in indexes
    ]
    positive_amounts = [value for value in positive_amounts if value > 0]
    return (
        float(min(positive_amounts) / max(positive_amounts))
        if positive_amounts and max(positive_amounts) > 0
        else 0.0
    )


def _member_total_stake_ratio(
    indexes: list[int] | tuple[int, ...],
    total_bets: np.ndarray,
) -> float:
    stakes = [float(total_bets[idx]) for idx in indexes if float(total_bets[idx]) > 0]
    return float(min(stakes) / max(stakes)) if stakes and max(stakes) > 0 else 0.0


def _rule_confidence(metrics: dict[str, float | int], cfg: CliqueRuleConfig) -> float:
    union_score = min(float(metrics["union_count"]) / max(float(cfg.strict_union_count), 1.0), 1.0)
    duplicate_score = 1.0 - min(
        float(metrics["duplicate_position_count"]) / max(float(metrics["union_count"]), 1.0),
        1.0,
    )
    ratio_score = min(float(metrics["avg_amount_ratio"]) / max(cfg.min_avg_amount_ratio, 1e-9), 1.0)
    total_stake_ratio_score = min(float(metrics["total_stake_ratio"]) / max(cfg.min_total_stake_ratio, 1e-9), 1.0)
    stake_score = min(float(metrics["group_total_stake"]) / max(cfg.min_group_total_stake, 1.0), 1.0)
    return float(
        0.35 * union_score
        + 0.20 * duplicate_score
        + 0.20 * ratio_score
        + 0.15 * total_stake_ratio_score
        + 0.10 * stake_score
    )
