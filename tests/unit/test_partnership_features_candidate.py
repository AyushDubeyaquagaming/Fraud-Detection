from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from fraud_detection.components.pair_scan import PairRuleConfig
from fraud_detection.components.partnership_features import (
    STAGE1_FEATURE_COLUMNS,
    compute_pair_rows_from_candidates,
    compute_partnership_features_from_candidates,
    ensure_stage1_schema,
    project_pair_scores_to_member_draw_rows,
)


def _candidate_row() -> dict:
    strict_a = [1] * 19 + [0] * 19
    strict_b = [0] * 19 + [1] * 19
    near = [0] * 19 + [1] * 18 + [0]
    ordinary = [1] + [0] * 37
    return {
        "draw_id": 9001,
        "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
        "trans_date_max": datetime(2026, 4, 27, 0, 1, tzinfo=timezone.utc),
        "qualifying_player_count": 4,
        "member_ids": ["A", "B", "C", "D"],
        "ccs_ids": ["CA", "CB", "CC", "CD"],
        "total_bet_amounts": [1000.0, 1000.0, 1000.0, 1000.0],
        "win_points": [1200.0, 1000.0, 1000.0, 1000.0],
        "coverage_bytes": [bytes(strict_a), bytes(strict_b), bytes(near), bytes(ordinary)],
        "amount_vector": [
            [10.0 if v else 0.0 for v in strict_a],
            [10.0 if v else 0.0 for v in strict_b],
            [10.0 if v else 0.0 for v in near],
            [10.0 if v else 0.0 for v in ordinary],
        ],
    }


def test_candidate_path_emits_strict_and_nearmiss_pairs() -> None:
    pair_df = compute_pair_rows_from_candidates(pd.DataFrame([_candidate_row()]), rule_config=PairRuleConfig())

    assert not pair_df.empty
    assert int(pair_df["is_strict_match"].sum()) == 1
    assert int(pair_df["is_nearmiss"].sum()) >= 1
    strict = pair_df.loc[pair_df["is_strict_match"].eq(1)].iloc[0]
    assert {strict["member_a"], strict["member_b"]} == {"A", "B"}


def test_negative_sampling_emits_sampled_ordinary_negatives() -> None:
    pair_df = compute_pair_rows_from_candidates(
        pd.DataFrame([_candidate_row()]),
        rule_config=PairRuleConfig(),
        mode="training",
        ordinary_negative_sample=2,
    )

    assert int(pair_df["sampled_negative"].sum()) == 2


def test_pair_to_member_projection_preserves_stage1_schema() -> None:
    stage1_df, pair_df, _ = compute_partnership_features_from_candidates(
        pd.DataFrame([_candidate_row()]),
        rule_config=PairRuleConfig(),
        mode="training",
        ordinary_negative_sample=1,
        rolling_context=False,
    )

    expected = ["member_id", "draw_id", "draw_date", "best_partner_member_id", *STAGE1_FEATURE_COLUMNS]
    assert list(stage1_df.columns) == expected
    assert not stage1_df.empty
    assert not pair_df.empty
    ensure_stage1_schema(stage1_df)


def test_projection_keeps_highest_risk_partner_per_member_draw() -> None:
    pair_df = pd.DataFrame(
        [
            {
                "draw_id": 1,
                "draw_date": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "member_a": "A",
                "member_b": "B",
                "coverage_count_a": 19,
                "coverage_count_b": 19,
                "overlap_count": 0,
                "union_count": 38,
                "union_coverage_pct": 1.0,
                "ratio_similarity": 1.0,
                "pair_net": 100.0,
                "stake_a": 1000.0,
                "stake_b": 1000.0,
                "win_points_a": 1100.0,
                "win_points_b": 1000.0,
                "is_strict_match": 1,
                "is_nearmiss": 0,
                "pair_risk_score": 1.0,
            },
            {
                "draw_id": 1,
                "draw_date": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "member_a": "A",
                "member_b": "C",
                "coverage_count_a": 19,
                "coverage_count_b": 18,
                "overlap_count": 1,
                "union_count": 36,
                "union_coverage_pct": 36 / 38,
                "ratio_similarity": 1.0,
                "pair_net": 100.0,
                "stake_a": 1000.0,
                "stake_b": 1000.0,
                "win_points_a": 1100.0,
                "win_points_b": 1000.0,
                "is_strict_match": 0,
                "is_nearmiss": 1,
                "pair_risk_score": 0.4,
            },
        ]
    )

    member_df = project_pair_scores_to_member_draw_rows(pair_df, rolling_context=False)
    a_row = member_df.loc[member_df["member_id"].eq("A")].iloc[0]

    assert a_row["best_partner_member_id"] == "B"
    assert a_row["n_low_overlap_partners_in_draw"] == 2
