from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from fraud_detection.components.pair_scan import (
    PairRuleConfig,
    build_draw_matrices,
    compute_pair_metrics,
    emit_pair_rows,
    per_position_ratio_similarity,
)


def _row(coverage: list[list[int]], amounts: list[list[float]], stakes=None, wins=None, ccs_ids=None):
    return {
        "draw_id": 1,
        "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
        "member_ids": [f"M{i}" for i in range(len(coverage))],
        "ccs_ids": ccs_ids or [f"CCS{i}" for i in range(len(coverage))],
        "total_bet_amounts": stakes or [1000.0] * len(coverage),
        "win_points": wins or [1000.0] * len(coverage),
        "coverage_bytes": [bytes(values) for values in coverage],
        "amount_vector": amounts,
    }


def test_strict_rule_fires_on_full_coverage_zero_overlap_positive_net() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0])

    rows = emit_pair_rows(row, PairRuleConfig(), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_strict_match"] == 1
    assert rows[0]["is_strict_collusion_pattern"] == 1
    assert rows[0]["bet_amount_ratio_within_10pct"] == 1
    assert rows[0]["union_count"] == 38
    assert rows[0]["overlap_count"] == 0


def test_37_of_38_is_nearmiss_not_strict() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 18 + [0]
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0])

    rows = emit_pair_rows(row, PairRuleConfig(), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_strict_match"] == 0
    assert rows[0]["is_nearmiss"] == 1
    assert rows[0]["union_count"] == 37
    assert rows[0]["ccs_a"] == "CCS0"
    assert rows[0]["ccs_b"] == "CCS1"
    assert rows[0]["pair_different_ccs"] == 1


def test_strict_same_ccs_pair_still_fires() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    row = _row(
        [left, right],
        [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
        ccs_ids=["CCS1", "CCS1"],
        wins=[1200.0, 1000.0],
    )

    rows = emit_pair_rows(row, PairRuleConfig(), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_strict_match"] == 1
    assert rows[0]["pair_different_ccs"] == 0


def test_nearmiss_same_ccs_filtered_when_required() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 18 + [0]
    row = _row(
        [left, right],
        [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
        ccs_ids=["CCS1", "CCS1"],
        wins=[1200.0, 1000.0],
    )

    assert emit_pair_rows(row, PairRuleConfig(nearmiss_require_different_ccs=True), mode="inference") == []


def test_nearmiss_same_ccs_fires_when_filter_disabled() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 18 + [0]
    row = _row(
        [left, right],
        [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
        ccs_ids=["CCS1", "CCS1"],
        wins=[1200.0, 1000.0],
    )

    rows = emit_pair_rows(row, PairRuleConfig(nearmiss_require_different_ccs=False), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_nearmiss"] == 1
    assert rows[0]["pair_different_ccs"] == 0


def test_nearmiss_different_ccs_still_fires() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 18 + [0]
    row = _row(
        [left, right],
        [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
        ccs_ids=["CCS1", "CCS2"],
        wins=[1200.0, 1000.0],
    )

    rows = emit_pair_rows(row, PairRuleConfig(nearmiss_require_different_ccs=True), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_nearmiss"] == 1
    assert rows[0]["pair_different_ccs"] == 1


def test_high_stake_cross_ccs_nearmiss_requires_similar_total_stake() -> None:
    left_positions = {1, 2, 4, 6, 8, 9, 10, 13, 15, 17, 19, 20, 22, 24, 26, 27, 28, 31, 33, 35, 36, 37}
    right_positions = {0, 3, 5, 7, 9, 11, 12, 14, 16, 18, 21, 23, 25, 27, 29, 30, 32, 34, 36}
    left = [int(idx in left_positions) for idx in range(38)]
    right = [int(idx in right_positions) for idx in range(38)]
    row = _row(
        [left, right],
        [[2000.0 if value else 0.0 for value in left], [2000.0 if value else 0.0 for value in right]],
        stakes=[45000.0, 37000.0],
        wins=[72000.0, 0.0],
        ccs_ids=["CCS016058", "CCS023061"],
    )

    rows = emit_pair_rows(
        row,
        PairRuleConfig(nearmiss_max_overlap=3, nearmiss_min_pair_net_per_stake=-0.15),
        mode="inference",
    )

    assert rows == []


def test_high_stake_cross_ccs_nearmiss_fires_when_total_stake_is_similar() -> None:
    left_positions = {1, 2, 4, 6, 8, 9, 10, 13, 15, 17, 19, 20, 22, 24, 26, 27, 28, 31, 33, 35, 36, 37}
    right_positions = {0, 3, 5, 7, 9, 11, 12, 14, 16, 18, 21, 23, 25, 27, 29, 30, 32, 34, 36}
    left = [int(idx in left_positions) for idx in range(38)]
    right = [int(idx in right_positions) for idx in range(38)]
    row = _row(
        [left, right],
        [[2000.0 if value else 0.0 for value in left], [2000.0 if value else 0.0 for value in right]],
        stakes=[45000.0, 41000.0],
        wins=[72000.0, 0.0],
        ccs_ids=["CCS016058", "CCS023061"],
    )

    rows = emit_pair_rows(
        row,
        PairRuleConfig(nearmiss_max_overlap=3, nearmiss_min_pair_net_per_stake=-0.20),
        mode="inference",
    )

    assert len(rows) == 1
    assert rows[0]["is_nearmiss"] == 1
    assert rows[0]["stake_ratio"] >= 0.90


def test_strict_allows_small_overlap_by_config() -> None:
    left = [1] * 20 + [0] * 18
    right = [1] + [0] * 19 + [1] * 18
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0])

    rows = emit_pair_rows(row, PairRuleConfig(), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_strict_match"] == 1
    assert rows[0]["is_min_overlap_pair"] == 1


def test_strict_does_not_fire_when_overlap_exceeds_config() -> None:
    left = [1] * 20 + [0] * 18
    right = [1] + [0] * 19 + [1] * 18
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0])

    rows = emit_pair_rows(row, PairRuleConfig(nearmiss_max_overlap=0), mode="inference")

    assert rows
    rows = emit_pair_rows(row, PairRuleConfig(strict_max_overlap_count=0, nearmiss_max_overlap=0), mode="inference")
    assert rows == []


def test_strict_does_not_fire_on_negative_pair_net() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[100.0, 100.0])

    assert emit_pair_rows(row, PairRuleConfig(), mode="inference") == []


def test_strict_rule_allows_small_pair_loss_for_exact_complementary_pattern() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    row = _row(
        [left, right],
        [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
        stakes=[19000.0, 19000.0],
        wins=[36000.0, 0.0],
    )

    rows = emit_pair_rows(row, PairRuleConfig(), mode="inference")

    assert len(rows) == 1
    assert rows[0]["is_strict_match"] == 1
    assert rows[0]["pair_net"] == -2000.0


def test_pair_direction_is_unique_and_self_pairs_ignored() -> None:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    rows = emit_pair_rows(_row([left, right], [[1.0 if v else 0.0 for v in left], [1.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0]))

    assert [(row["member_a"], row["member_b"]) for row in rows] == [("M0", "M1")]


def test_ratio_similarity_correctness() -> None:
    amounts = np.array([[10.0, 20.0, 0.0], [5.0, 20.0, 5.0]])
    coverage = amounts > 0

    ratio = per_position_ratio_similarity(amounts, coverage)

    assert np.isclose(ratio[0, 1], 10.0 / 15.0)


def test_empty_and_single_player_draw_safe() -> None:
    assert emit_pair_rows(_row([], [])) == []
    assert emit_pair_rows(_row([[1] * 38], [[1.0] * 38])) == []


def test_symmetric_metrics_hold() -> None:
    coverage, amounts = build_draw_matrices([bytes([1, 0, 1]), bytes([1, 1, 0])], [[1, 0, 1], [1, 1, 0]], board_size=3)
    metrics = compute_pair_metrics(coverage, amounts, np.array([1000.0, 1000.0]), np.array([1200.0, 900.0]))

    assert metrics["overlap_count"][0, 1] == metrics["overlap_count"][1, 0]
    assert metrics["union_count"][0, 1] == metrics["union_count"][1, 0]
    assert metrics["ratio_similarity"][0, 1] == metrics["ratio_similarity"][1, 0]
