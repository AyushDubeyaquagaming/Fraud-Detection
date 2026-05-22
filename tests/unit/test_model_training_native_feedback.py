from __future__ import annotations

import pandas as pd

from fraud_detection.components.model_training import _native_feedback_member_labels


def test_native_feedback_member_labels_collects_fraud_members_and_weights():
    pair_scored = pd.DataFrame(
        [
            {
                "member_a": "member_a",
                "member_b": "member_b",
                "native_member_a_label": "fraud",
                "native_member_b_label": "not_fraud",
                "native_member_a_weight": 4.0,
                "native_member_b_weight": 3.0,
            },
            {
                "member_a": "member_c",
                "member_b": "member_d",
                "native_member_a_label": None,
                "native_member_b_label": "fraud",
                "native_member_a_weight": 1.0,
                "native_member_b_weight": 2.5,
            },
        ]
    )

    fraud_members, member_weights = _native_feedback_member_labels(
        pair_scored,
        {"native_feedback": {"confirmed_fraud_weight": 2.0, "confirmed_not_fraud_weight": 2.0}},
    )

    assert fraud_members == {"MEMBER_A", "MEMBER_D"}
    assert member_weights == {"MEMBER_A": 4.0, "MEMBER_B": 3.0, "MEMBER_D": 2.5}
