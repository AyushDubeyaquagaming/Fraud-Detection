from __future__ import annotations

import pandas as pd

from fraud_detection.components.analyst_label_overlay import apply_pair_label_overrides


def _pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draw_id": [1],
            "member_a": ["A"],
            "member_b": ["B"],
            "is_strict_match": [1],
            "is_nearmiss": [0],
            "sampled_negative": [0],
            "label_stage1": [1],
            "label_gold": [0],
            "label_source": ["strict_rule"],
            "sample_weight": [1.0],
        }
    )


def test_both_fraud_members_create_positive_override():
    result, summary = apply_pair_label_overrides(
        _pair().assign(label_stage1=[0]),
        [
            {"draw_id": 1, "member_id": "A", "label": "fraud"},
            {"draw_id": 1, "member_id": "B", "label": "fraud"},
        ],
    )

    assert result["label_stage1"].iloc[0] == 1
    assert result["label_gold"].iloc[0] == 1
    assert result["label_source"].iloc[0] == "derived_pair_analyst"
    assert summary["positive_overrides"] == 1


def test_both_not_fraud_members_create_negative_override():
    result, summary = apply_pair_label_overrides(
        _pair(),
        [
            {"draw_id": 1, "member_id": "A", "label": "not_fraud"},
            {"draw_id": 1, "member_id": "B", "label": "not_fraud"},
        ],
    )

    assert result["label_stage1"].iloc[0] == 0
    assert result["label_gold"].iloc[0] == 0
    assert result["label_source"].iloc[0] == "analyst_not_fraud_pair"
    assert summary["negative_overrides"] == 1


def test_single_member_label_does_not_override():
    result, summary = apply_pair_label_overrides(
        _pair(),
        [{"draw_id": 1, "member_id": "A", "label": "not_fraud"}],
    )

    assert result["label_stage1"].iloc[0] == 1
    assert summary["negative_overrides"] == 0


def test_conflicting_member_labels_do_not_override():
    result, summary = apply_pair_label_overrides(
        _pair(),
        [
            {"draw_id": 1, "member_id": "A", "label": "fraud"},
            {"draw_id": 1, "member_id": "B", "label": "not_fraud"},
        ],
    )

    assert result["label_stage1"].iloc[0] == 1
    assert result["label_source"].iloc[0] == "strict_rule"
    assert summary["conflicts"] == 1


def test_unrelated_draw_labels_do_not_leak():
    result, summary = apply_pair_label_overrides(
        _pair(),
        [
            {"draw_id": 2, "member_id": "A", "label": "not_fraud"},
            {"draw_id": 2, "member_id": "B", "label": "not_fraud"},
        ],
    )

    assert result["label_stage1"].iloc[0] == 1
    assert summary["negative_overrides"] == 0
