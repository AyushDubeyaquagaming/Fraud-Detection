from __future__ import annotations

import pandas as pd

from fraud_detection.utils.fraud_label_matching import (
    DROPPED_KEY_MISMATCH,
    DROPPED_NO_HISTORY,
    DROPPED_NOT_IN_SOURCE,
    DROPPED_OUT_OF_WINDOW,
    MATCHED,
    classify_fraud_csv,
    summarize_verdicts,
)


def test_classify_fraud_csv_reports_stable_statuses() -> None:
    fraud = pd.DataFrame(
        [
            {"member_id": "a", "draw_id": 10, "date": "2026-01-03"},
            {"member_id": "b", "draw_id": 20, "date": "2026-01-03"},
            {"member_id": "c", "draw_id": 30, "date": "2026-01-03"},
            {"member_id": "d", "draw_id": 40, "date": "2026-01-03"},
            {"member_id": "e", "draw_id": 50, "date": "2026-02-01"},
        ]
    )
    available = pd.DataFrame(
        [
            {"member_id": "A", "draw_id": 1, "ts": "2026-01-01"},
            {"member_id": "A", "draw_id": 10, "ts": "2026-01-03"},
            {"member_id": "B", "draw_id": 20, "ts": "2026-01-03"},
            {"member_id": "D", "draw_id": 41, "ts": "2026-01-03"},
            {"member_id": "E", "draw_id": 51, "ts": "2026-01-01"},
        ]
    )

    verdicts = classify_fraud_csv(fraud, available)
    by_member = dict(zip(verdicts["fraud_csv_member_id"], verdicts["final_status"]))

    assert by_member["A"] == MATCHED
    assert by_member["B"] == DROPPED_NO_HISTORY
    assert by_member["C"] == DROPPED_NOT_IN_SOURCE
    assert by_member["D"] == DROPPED_KEY_MISMATCH
    assert by_member["E"] == DROPPED_OUT_OF_WINDOW
    assert summarize_verdicts(verdicts)["event_status_counts"][MATCHED] == 1
