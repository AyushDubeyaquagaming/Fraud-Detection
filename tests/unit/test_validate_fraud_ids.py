from __future__ import annotations

import pandas as pd

from scripts.validate_fraud_ids import _build_report, _normalize_ccs_id, _verdict


def test_normalize_ccs_id_matches_mongo_and_config_formats():
    assert _normalize_ccs_id("38767") == "38767"
    assert _normalize_ccs_id("CCS038767") == "38767"
    assert _normalize_ccs_id(" ccs-000123 ") == "123"


def test_verdict_requires_draw_and_ccs_match():
    base = {
        "found_in_mongo": True,
        "draw_id_present": True,
        "ccs_id_match": True,
        "pre_fraud_draw_count": 1,
    }
    assert _verdict(base, min_pre_fraud_draws=1) == "add"

    assert _verdict({**base, "draw_id_present": False}, min_pre_fraud_draws=1) == "needs_review"
    assert _verdict({**base, "ccs_id_match": False}, min_pre_fraud_draws=1) == "needs_review"
    assert _verdict({**base, "ccs_id_match": None}, min_pre_fraud_draws=1) == "needs_review"
    assert _verdict({**base, "pre_fraud_draw_count": 0}, min_pre_fraud_draws=1) == "needs_more_history"
    assert _verdict({**base, "found_in_mongo": False}, min_pre_fraud_draws=1) == "not_found"


def test_build_report_treats_numeric_and_prefixed_ccs_as_match():
    fraud_date = pd.Timestamp("2026-04-21T00:00:00Z")
    pending = [
        {
            "member_id": "GK001",
            "ccs_id": "38767",
            "draw_id": 123,
            "fraud_date": fraud_date,
        }
    ]
    transactions = pd.DataFrame(
        [
            {
                "member_id_norm": "GK001",
                "ccs_id": "CCS038767",
                "draw_id_norm": 123,
                "ts": pd.Timestamp("2026-04-18T13:07:45Z"),
            }
        ]
    )

    rows = _build_report(
        pending=pending,
        transactions=transactions,
        window_days=7,
        today=pd.Timestamp("2026-04-29T00:00:00Z"),
        lookback_days=90,
        min_pre_fraud_draws=1,
    )

    assert rows[0]["ccs_id_match"] is True
    assert rows[0]["ccs_id_normalized"] == "38767"
    assert rows[0]["verdict"] == "add"
