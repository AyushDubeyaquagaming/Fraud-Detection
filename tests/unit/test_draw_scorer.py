from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS
from fraud_detection.components.partnership_features import ROULETTE_POSITIONS, STAGE1_FEATURE_COLUMNS, STAGE2_FEATURE_COLUMNS
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer


def _bets(positions: list[str], amount: float = 100.0) -> str:
    return json.dumps([{"number": position, "bet_amount": amount} for position in positions])


def test_draw_scorer_returns_partnership_document():
    raw = pd.DataFrame(
        [
            {
                "draw_id": 99,
                "member_id": "A",
                "bets": _bets(ROULETTE_POSITIONS[:19]),
                "total_bet_amount": 1900.0,
                "win_points": 2500.0,
                "trans_date": pd.Timestamp("2026-04-21T10:00:00Z"),
            },
            {
                "draw_id": 99,
                "member_id": "B",
                "bets": _bets(ROULETTE_POSITIONS[19:]),
                "total_bet_amount": 1900.0,
                "win_points": 2500.0,
                "trans_date": pd.Timestamp("2026-04-21T10:00:01Z"),
            },
        ]
    )
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    # Fit once so predict_proba/classes_ are available.
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE1_FEATURE_COLUMNS}), [1, 1])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [1, 1])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage1_feature_columns": STAGE1_FEATURE_COLUMNS,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "stage1_high_threshold": 0.5,
        "stage2_alert_threshold": 0.5,
    }

    result = DrawScorer(bundle, source_run_id="run_test").score_draw(raw)

    assert result.draw_id == 99
    assert result.requires_review is True
    assert {member["member_id"] for member in result.flagged_members} == {"A", "B"}
    assert {member["win_amount"] for member in result.flagged_members} == {2500.0}
    assert {member["bet_amount"] for member in result.flagged_members} == {1900.0}
    assert not any(member["high_amount_flag"] for member in result.flagged_members)
    assert result.partnerships
    assert result.partnerships[0]["union_coverage"] == 1.0


def test_candidate_draw_scorer_defaults_to_pair_stage1_columns() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [1, 1])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [1, 1])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "use_candidate_store": True,
    }

    result = DrawScorer(bundle, source_run_id="run_test").score_candidate_draw(
        {
            "draw_id": 101,
            "qualifying_player_count": 2,
            "member_ids": ["A", "B"],
            "ccs_ids": ["CA", "CB"],
            "trans_date_min": pd.Timestamp("2026-04-21T10:00:00Z"),
            "trans_date_max": pd.Timestamp("2026-04-21T10:00:01Z"),
            "coverage_bytes": [bytes([1] * 19 + [0] * 19), bytes([0] * 19 + [1] * 19)],
            "amount_vector": [[10.0] * 19 + [0.0] * 19, [0.0] * 19 + [10.0] * 19],
            "total_bet_amounts": [1900.0, 1900.0],
            "win_points": [2500.0, 2500.0],
        }
    )

    assert result.draw_id == 101
    assert result.partnerships
    assert {member["win_amount"] for member in result.flagged_members} == {2500.0}
    assert {member["bet_amount"] for member in result.flagged_members} == {1900.0}
    assert "no_member_history" in result.response_details


def test_candidate_batch_scoring_matches_single_draw_path() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [1, 1])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [1, 1])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "use_candidate_store": True,
    }
    candidate = {
        "draw_id": 202,
        "qualifying_player_count": 2,
        "member_ids": ["A", "B"],
        "ccs_ids": ["CA", "CB"],
        "trans_date_min": pd.Timestamp("2026-04-21T10:00:00Z"),
        "trans_date_max": pd.Timestamp("2026-04-21T10:00:01Z"),
        "coverage_bytes": [bytes([1] * 19 + [0] * 19), bytes([0] * 19 + [1] * 19)],
        "amount_vector": [[10.0] * 19 + [0.0] * 19, [0.0] * 19 + [10.0] * 19],
        "total_bet_amounts": [1900.0, 1900.0],
        "win_points": [2500.0, 2500.0],
    }

    scorer = DrawScorer(bundle, source_run_id="run_test")
    single = scorer.score_candidate_draw(candidate).to_mongo_doc()
    batched = scorer.score_candidate_batch(pd.DataFrame([candidate]))[0].to_mongo_doc()

    for volatile in ("scored_at",):
        single.pop(volatile)
        batched.pop(volatile)
    assert batched == single
