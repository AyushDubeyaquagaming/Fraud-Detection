from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fraud_detection.serving.routes.live_scoring import (
    _apply_known_fraud_overlay,
    _candidate_rows_for_draw,
    _candidate_rows_for_member,
    _member_evidence,
)
from fraud_detection.serving.schemas import DrawPayloadScoreRequest


def test_member_evidence_returns_partner_draw_context():
    doc = {
        "draw_id": 123,
        "partnerships": [
            {
                "member_ids": ["A", "B"],
                "union_coverage": 1.0,
            }
        ],
        "flagged_members": [
            {
                "member_id": "A",
                "stage1_score_in_draw": 0.95,
                "stage2_score": 0.80,
                "best_partner_member_id": "B",
            }
        ],
    }

    evidence = _member_evidence(doc, "A")

    assert evidence is not None
    assert evidence.draw_id == 123
    assert evidence.score_reason == "partnership_pattern"
    assert evidence.partner_member_ids == ["B"]
    assert evidence.max_union_coverage == 1.0


def test_full_payload_request_is_separate_compatibility_schema():
    request = DrawPayloadScoreRequest(
        draw_id=1,
        players=[
            {
                "member_id": "A",
                "total_bet_amount": 1000.0,
                "bets": [{"number": "1", "bet_amount": 1000.0}],
            }
        ],
    )

    assert request.players[0].member_id == "A"


def test_candidate_rows_for_member_reads_partitioned_store(tmp_path):
    store = tmp_path / "candidate_draws" / "year=2026" / "month=05" / "week=18"
    store.mkdir(parents=True)
    pd.DataFrame(
        {
            "draw_id": [7297697, 7297698],
            "member_ids": [["GK00236424", "GK00511854"], ["GK00000001"]],
            "trans_date_min": pd.to_datetime(["2026-05-03T04:07:47Z", "2026-05-03T05:00:00Z"]),
        }
    ).to_parquet(store / "draws.parquet", index=False)
    context = SimpleNamespace(model_bundle={"candidate_store_path": str(tmp_path / "candidate_draws")})

    rows = _candidate_rows_for_member("GK00236424", context=context, max_draws=10)

    assert rows["draw_id"].tolist() == [7297697]

    draw_rows = _candidate_rows_for_draw(7297697, context=context)
    assert list(draw_rows["member_ids"].iloc[0]) == ["GK00236424", "GK00511854"]


def test_known_fraud_overlay_surfaces_labeled_pair():
    doc = {
        "draw_id": 7242014,
        "partnerships": [],
        "flagged_members": [],
        "max_stage1_score": 0.0,
        "max_stage2_score": 0.0,
        "requires_review": False,
        "response_details": ["no_member_history"],
    }
    row = {
        "draw_id": 7242014,
        "member_ids": ["GK00116069", "GK00123072"],
        "coverage_bytes": [bytes([1] * 38), bytes([1] * 38)],
        "amount_vector": [[1.0] * 38, [1.0] * 38],
        "total_bet_amounts": [45000.0, 37000.0],
        "win_points": [72000.0, 0.0],
    }

    result = _apply_known_fraud_overlay(doc, row)

    assert result["requires_review"] is True
    assert result["max_stage1_score"] == 1.0
    assert result["partnerships"][0]["member_ids"] == ["GK00116069", "GK00123072"]
    assert sorted(item["member_id"] for item in result["flagged_members"]) == ["GK00116069", "GK00123072"]
    first = next(item for item in result["flagged_members"] if item["member_id"] == "GK00116069")
    assert first["bet_amount"] == 45000.0
    assert first["win_amount"] == 72000.0
    assert first["high_amount_flag"] is True
    assert "known_fraud_label_overlay" in result["response_details"]
