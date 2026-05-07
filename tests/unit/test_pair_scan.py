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


def _row(coverage: list[list[int]], amounts: list[list[float]], stakes=None, wins=None):
    return {
        "draw_id": 1,
        "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
        "member_ids": [f"M{i}" for i in range(len(coverage))],
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


def test_strict_does_not_fire_on_overlap() -> None:
    left = [1] * 20 + [0] * 18
    right = [1] + [0] * 19 + [1] * 18
    row = _row([left, right], [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]], wins=[1200.0, 1000.0])

    rows = emit_pair_rows(row, PairRuleConfig(nearmiss_max_overlap=0), mode="inference")

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
