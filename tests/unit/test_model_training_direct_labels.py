from __future__ import annotations

from datetime import datetime, timezone

from fraud_detection.components.model_training import _direct_analyst_fraud_members
from fraud_detection.utils import mongo_predictions


def test_direct_analyst_fraud_members_reads_latest_scoped_member_labels(monkeypatch):
    labels = [
        {
            "draw_id": 1,
            "member_id": "member_a",
            "label": "not_fraud",
            "decided_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        },
        {
            "draw_id": 1,
            "member_id": "member_a",
            "label": "fraud",
            "decided_at": datetime(2026, 5, 2, tzinfo=timezone.utc),
        },
        {
            "draw_id": 2,
            "member_id": "member_b",
            "label": "not_fraud",
            "decided_at": datetime(2026, 5, 2, tzinfo=timezone.utc),
        },
    ]

    def fake_read(draw_ids):
        assert draw_ids == [1, 2]
        return labels

    monkeypatch.setattr(mongo_predictions, "read_analyst_labels_for_draws", fake_read)

    assert _direct_analyst_fraud_members([1, 2, 2]) == {"MEMBER_A"}
