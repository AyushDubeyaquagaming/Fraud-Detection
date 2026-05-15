from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
import pandas as pd

from fraud_detection.serving.routes import live_scoring
from fraud_detection.serving.routes.live_scoring import (
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


def test_label_route_is_not_registered():
    app = FastAPI()
    app.include_router(live_scoring.router)

    paths = {route.path for route in app.routes}

    assert "/label" not in paths


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


def test_score_alerts_includes_ccs_context_for_flagged_members(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    candidate_row = {
        "draw_id": 1,
        "member_ids": ["A", "B"],
        "ccs_ids": ["CA", "CB"],
        "trans_date_min": now - pd.Timedelta(days=1),
    }
    result_doc = {
        "draw_id": 1,
        "requires_review": True,
        "partnerships": [{"member_ids": ["A", "B"], "union_coverage": 1.0}],
        "flagged_members": [
            {
                "member_id": "A",
                "stage1_score_in_draw": 1.0,
                "stage2_score": 0.0,
                "best_partner_member_id": "B",
                "bet_amount": 12000.0,
                "win_amount": 30000.0,
                "high_amount_flag": True,
            }
        ],
        "max_stage1_score": 1.0,
        "max_stage2_score": 0.0,
        "response_details": ["no_member_history"],
    }
    context = SimpleNamespace(
        model_bundle={"stage1_model": object(), "stage2_model": object()},
        source_run_id="run_test",
        partnership_table=None,
        ccs_concentration_table=None,
        evaluation_metadata={"label_status": "unavailable"},
    )
    monkeypatch.setattr(live_scoring, "_prediction_backfill_docs", lambda *_args, **_kwargs: [result_doc])
    monkeypatch.setattr(live_scoring, "_candidate_row_for_doc", lambda *_args, **_kwargs: candidate_row)

    response = live_scoring.score_alerts(context=context)

    member = response.alerts[0].flagged_members[0]
    assert member.ccs_id == "CA"
    assert member.bet_amount == 12000.0
    assert member.win_amount == 30000.0
    assert response.alerts[0].response_details == ["no_member_history", "stage2_unavailable_no_labels"]


def test_score_alerts_applies_window_and_max_draws_before_risk_sort(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    docs = [
        {
            "draw_id": 1,
            "requires_review": True,
            "partnerships": [{"member_ids": ["A", "B"], "union_coverage": 1.0}],
            "flagged_members": [{"member_id": "A", "stage1_score_in_draw": 1.0, "win_amount": 99999.0}],
            "max_stage1_score": 1.0,
        },
        {
            "draw_id": 2,
            "requires_review": True,
            "partnerships": [{"member_ids": ["C", "D"], "union_coverage": 1.0}],
            "flagged_members": [{"member_id": "C", "stage1_score_in_draw": 1.0, "win_amount": 100.0}],
            "max_stage1_score": 1.0,
        },
        {
            "draw_id": 3,
            "requires_review": True,
            "partnerships": [{"member_ids": ["E", "F"], "union_coverage": 1.0}],
            "flagged_members": [{"member_id": "E", "stage1_score_in_draw": 1.0, "win_amount": 10.0}],
            "max_stage1_score": 1.0,
        },
    ]
    rows = {
        1: {"draw_id": 1, "member_ids": ["A", "B"], "ccs_ids": ["CA", "CB"], "trans_date_min": now - pd.Timedelta(days=20)},
        2: {"draw_id": 2, "member_ids": ["C", "D"], "ccs_ids": ["CC", "CD"], "trans_date_min": now - pd.Timedelta(days=2)},
        3: {"draw_id": 3, "member_ids": ["E", "F"], "ccs_ids": ["CE", "CF"], "trans_date_min": now - pd.Timedelta(days=1)},
    }
    context = SimpleNamespace(
        model_bundle={"stage1_model": object(), "stage2_model": object()},
        source_run_id="run_test",
        partnership_table=None,
        ccs_concentration_table=None,
        evaluation_metadata={"label_status": "available"},
    )
    monkeypatch.setattr(live_scoring, "_prediction_backfill_docs", lambda *_args, **_kwargs: docs)
    monkeypatch.setattr(live_scoring, "_candidate_row_for_doc", lambda draw_id, _context: rows[int(draw_id)])

    response = live_scoring.score_alerts(lookback_days=7, max_draws=1, limit=10, context=context)

    assert response.draws_scanned == 1
    assert [alert.draw_id for alert in response.alerts] == [3]


def test_score_ccs_groups_alert_queue_from_backfill(monkeypatch):
    context = SimpleNamespace(
        model_bundle={"stage1_model": object(), "stage2_model": object()},
        source_run_id="run_test",
        partnership_table=None,
        ccs_concentration_table=None,
        evaluation_metadata={"label_status": "unavailable"},
    )
    alert = live_scoring.AlertDraw(
        draw_id=7,
        draw_date="2026-05-07T00:00:00+00:00",
        risk_tier="HIGH",
        partnership_count=1,
        flagged_member_count=1,
        flagged_member_ids=["A"],
        flaggedMembers=[
            live_scoring.AlertFlaggedMember(
                member_id="A",
                ccsId="CA",
                stage1_score_in_draw=0.7,
                best_partner_member_id="B",
                betAmount=1000.0,
                winAmount=5000.0,
                highAmountFlag=False,
                partner_member_ids=["B"],
            )
        ],
        ccs_ids=["CA"],
        highAmountMemberCount=0,
        maxBetAmount=1000.0,
        maxWinAmount=5000.0,
        max_stage1_score=0.7,
        response_details=["stage2_unavailable_no_labels"],
    )
    monkeypatch.setattr(live_scoring, "_build_alert_draws_from_backfill", lambda **kwargs: ([alert], 1))

    response = live_scoring.score_ccs(
        live_scoring.CcsScoreRequest(ccs_ids=["CA"], lookback_days=7, max_draws=10000),
        context=context,
    )

    assert response.draws_scanned == 1
    assert response.ccs_scores[0].ccs_id == "CA"
    assert response.ccs_scores[0].flagged_members == ["A"]
    assert response.ccs_scores[0].evidence[0]["draw_id"] == 7


def test_score_ccs_filters_requested_ccs_before_max_draw_limit(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    docs = [
        {
            "draw_id": 1,
            "requires_review": True,
            "partnerships": [{"member_ids": ["A", "B"], "union_coverage": 1.0}],
            "flagged_members": [{"member_id": "A", "stage1_score_in_draw": 1.0, "win_amount": 50000.0}],
            "max_stage1_score": 1.0,
        },
        {
            "draw_id": 2,
            "requires_review": True,
            "partnerships": [{"member_ids": ["C", "D"], "union_coverage": 1.0}],
            "flagged_members": [{"member_id": "C", "stage1_score_in_draw": 1.0, "win_amount": 100.0}],
            "max_stage1_score": 1.0,
        },
    ]
    rows = {
        1: {"draw_id": 1, "member_ids": ["A", "B"], "ccs_ids": ["OTHER", "OTHER2"], "trans_date_min": now},
        2: {"draw_id": 2, "member_ids": ["C", "D"], "ccs_ids": ["TARGET", "TARGET2"], "trans_date_min": now - pd.Timedelta(hours=1)},
    }
    context = SimpleNamespace(
        model_bundle={"stage1_model": object(), "stage2_model": object()},
        source_run_id="run_test",
        partnership_table=None,
        ccs_concentration_table=None,
        evaluation_metadata={"label_status": "available"},
    )
    monkeypatch.setattr(live_scoring, "_prediction_backfill_docs", lambda *_args, **_kwargs: docs)
    monkeypatch.setattr(live_scoring, "_candidate_row_for_doc", lambda draw_id, _context: rows[int(draw_id)])

    response = live_scoring.score_ccs(
        live_scoring.CcsScoreRequest(ccs_ids=["TARGET"], lookback_days=7, max_draws=1),
        context=context,
    )

    assert response.draws_scanned == 1
    assert response.ccs_scores[0].ccs_id == "TARGET"
    assert response.ccs_scores[0].risk_tier == "HIGH"
    assert response.ccs_scores[0].evidence_draw_ids == [2]
