from __future__ import annotations

import pandas as pd

from fraud_detection.components.clique_scan import CliqueRuleConfig, emit_clique_rows
from fraud_detection.components.partnership_features import (
    compute_clique_rows_from_candidates,
    compute_pair_rows_from_candidates,
    project_clique_scores_to_member_draw_rows,
)


def _candidate_frame() -> pd.DataFrame:
    coverage = [
        [1] * 12 + [0] * 26,
        [0] * 12 + [1] * 12 + [0] * 14,
        [0] * 24 + [1] * 12 + [0] * 2,
    ]
    amounts = [[5000.0 if value else 0.0 for value in row] for row in coverage]
    return pd.DataFrame(
        [
            {
                "draw_id": 7305365,
                "trans_date_min": pd.Timestamp("2026-05-08T11:56:36Z"),
                "trans_date_max": pd.Timestamp("2026-05-08T11:57:25Z"),
                "member_ids": ["GK00527181", "GK00140513", "GK00527185"],
                "ccs_ids": ["CCS040502", "CCS040502", "CCS040502"],
                "coverage_bytes": [bytes(row) for row in coverage],
                "amount_vector": amounts,
                "total_bet_amounts": [60000.0, 60000.0, 60000.0],
                "win_points": [0.0, 0.0, 0.0],
            }
        ]
    )


def _section_team_candidate_frame() -> pd.DataFrame:
    sections = [
        (["GK00527181", "GK00140513", "GK00527185"], "CCS040502", list(range(14, 26)), 60000.0, 0.0),
        (["GK00537590", "GK00141731", "GK00537586"], "CCS041720", list(range(2, 14)), 60000.0, 180000.0),
        (["GK00301706", "GK00139170", "GK00301710"], "CCS039159", list(range(26, 38)), 60000.0, 0.0),
        (["GK00527633", "GK00527634", "GK00140526"], "CCS040515", [0, 1], 10000.0, 0.0),
    ]
    member_ids: list[str] = []
    ccs_ids: list[str] = []
    coverage: list[list[int]] = []
    amounts: list[list[float]] = []
    stakes: list[float] = []
    wins: list[float] = []
    for members, ccs_id, positions, stake, win_points in sections:
        mask = [1 if idx in positions else 0 for idx in range(38)]
        amount = stake / len(positions)
        for member in members:
            member_ids.append(member)
            ccs_ids.append(ccs_id)
            coverage.append(mask)
            amounts.append([amount if value else 0.0 for value in mask])
            stakes.append(stake)
            wins.append(win_points)
    return pd.DataFrame(
        [
            {
                "draw_id": 7305365,
                "trans_date_min": pd.Timestamp("2026-05-08T11:56:36Z"),
                "trans_date_max": pd.Timestamp("2026-05-08T11:57:25Z"),
                "member_ids": member_ids,
                "ccs_ids": ccs_ids,
                "coverage_bytes": [bytes(row) for row in coverage],
                "amount_vector": amounts,
                "total_bet_amounts": stakes,
                "win_points": wins,
            }
        ]
    )


def test_same_ccs_three_member_board_partition_is_a_strict_clique() -> None:
    row = _candidate_frame().iloc[0].to_dict()

    events = emit_clique_rows(row, CliqueRuleConfig())

    assert len(events) == 1
    assert set(events[0]["member_ids"]) == {"GK00527181", "GK00140513", "GK00527185"}
    assert events[0]["ccs_id"] == "CCS040502"
    assert events[0]["clique_size"] == 3
    assert events[0]["union_count"] == 36
    assert events[0]["duplicate_position_count"] == 0
    assert events[0]["total_stake_ratio"] == 1.0
    assert events[0]["is_strict_clique"] == 1
    assert events[0]["clique_risk_score"] == 1.0


def test_same_ccs_clique_requires_similar_total_stake() -> None:
    row = _candidate_frame().iloc[0].to_dict()
    row["total_bet_amounts"] = [60000.0, 60000.0, 30000.0]

    events = emit_clique_rows(row, CliqueRuleConfig())

    assert events == []


def test_clique_projection_flags_members_when_pair_scan_has_no_match() -> None:
    candidates = _candidate_frame()

    pair_events = compute_pair_rows_from_candidates(candidates)
    clique_events = compute_clique_rows_from_candidates(candidates)
    stage1 = project_clique_scores_to_member_draw_rows(clique_events, candidates, rolling_context=False)

    assert pair_events.empty
    assert len(clique_events) == 1
    assert set(stage1["member_id"]) == {"GK00527181", "GK00140513", "GK00527185"}
    assert set(stage1["stage1_score"]) == {1.0}
    assert set(stage1["n_strict_pairs_in_draw"]) == {2}


def test_section_team_duplicate_clusters_covering_board_are_flagged() -> None:
    candidates = _section_team_candidate_frame()

    pair_events = compute_pair_rows_from_candidates(candidates)
    clique_events = compute_clique_rows_from_candidates(candidates)
    stage1 = project_clique_scores_to_member_draw_rows(clique_events, candidates, rolling_context=False)

    assert pair_events.empty
    assert len(clique_events) == 1
    assert clique_events.iloc[0]["pattern_type"] == "section_team"
    assert clique_events.iloc[0]["cluster_count"] == 4
    assert clique_events.iloc[0]["union_count"] == 38
    assert clique_events.iloc[0]["total_stake_ratio"] == 1.0
    assert clique_events.iloc[0]["is_strict_clique"] == 1
    assert set(stage1["member_id"]) == set(candidates.iloc[0]["member_ids"])
    assert set(stage1["stage1_score"]) == {1.0}
