from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fraud_detection.components.ccs_features import CCS_FEATURE_COLUMNS
from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS
from fraud_detection.components.partnership_features import ROULETTE_POSITIONS, STAGE1_FEATURE_COLUMNS, STAGE2_FEATURE_COLUMNS
from fraud_detection.serving.live_scoring import draw_scorer as draw_scorer_module
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


def test_candidate_nearmiss_gets_rule_floor_when_model_score_is_low() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [0, 0])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [0, 0])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "use_candidate_store": True,
        "stage1_flag_threshold": 0.7,
    }

    result = DrawScorer(bundle, source_run_id="run_test").score_candidate_draw(
        {
            "draw_id": 102,
            "qualifying_player_count": 2,
            "member_ids": ["A", "B"],
            "ccs_ids": ["CA", "CB"],
            "trans_date_min": pd.Timestamp("2026-04-21T10:00:00Z"),
            "trans_date_max": pd.Timestamp("2026-04-21T10:00:01Z"),
            "coverage_bytes": [bytes([1] * 19 + [0] * 19), bytes([0] * 19 + [1] * 18 + [0])],
            "amount_vector": [[10.0] * 19 + [0.0] * 19, [0.0] * 19 + [10.0] * 18 + [0.0]],
            "total_bet_amounts": [1900.0, 1800.0],
            "win_points": [2000.0, 1900.0],
        }
    )

    assert result.requires_review is True
    assert result.max_stage1_score == 0.7
    assert {member["stage1_score_in_draw"] for member in result.flagged_members} == {0.7}
    assert result.partnerships[0]["is_section_a"] is True


def test_candidate_exact_partnership_high_amount_regression() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [0, 0])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [0, 0])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "use_candidate_store": True,
        "stage1_flag_threshold": 0.7,
    }

    result = DrawScorer(bundle, source_run_id="run_test").score_candidate_draw(
        {
            "draw_id": 7300000,
            "qualifying_player_count": 2,
            "member_ids": ["SYNTH_EXACT_A", "SYNTH_EXACT_B"],
            "ccs_ids": ["SYNTH_CCS", "SYNTH_CCS"],
            "trans_date_min": pd.Timestamp("2026-05-10T18:27:57Z"),
            "trans_date_max": pd.Timestamp("2026-05-10T18:28:00Z"),
            "coverage_bytes": [bytes([1] * 19 + [0] * 19), bytes([0] * 19 + [1] * 19)],
            "amount_vector": [[5000.0] * 19 + [0.0] * 19, [0.0] * 19 + [5000.0] * 19],
            "total_bet_amounts": [95000.0, 95000.0],
            "win_points": [90000.0, 90000.0],
        }
    )

    assert result.requires_review is True
    assert result.max_stage1_score == 1.0
    assert {member["member_id"] for member in result.flagged_members} == {"SYNTH_EXACT_A", "SYNTH_EXACT_B"}
    assert {member["bet_amount"] for member in result.flagged_members} == {95000.0}
    assert all(member["high_amount_flag"] for member in result.flagged_members)
    assert result.partnerships[0]["is_section_b"] is True


def test_candidate_three_member_same_ccs_clique_flags_draw_7305365_pattern() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [0, 0])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [0, 0])
    bundle = {
        "model_version": "partnership_v1",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "use_candidate_store": True,
        "stage1_flag_threshold": 0.7,
    }
    coverage = [
        [1] * 12 + [0] * 26,
        [0] * 12 + [1] * 12 + [0] * 14,
        [0] * 24 + [1] * 12 + [0] * 2,
    ]

    result = DrawScorer(bundle, source_run_id="run_test").score_candidate_draw(
        {
            "draw_id": 7305365,
            "qualifying_player_count": 3,
            "member_ids": ["GK00527181", "GK00140513", "GK00527185"],
            "ccs_ids": ["CCS040502", "CCS040502", "CCS040502"],
            "trans_date_min": pd.Timestamp("2026-05-08T11:56:36Z"),
            "trans_date_max": pd.Timestamp("2026-05-08T11:57:25Z"),
            "coverage_bytes": [bytes(row) for row in coverage],
            "amount_vector": [[5000.0 if value else 0.0 for value in row] for row in coverage],
            "total_bet_amounts": [60000.0, 60000.0, 60000.0],
            "win_points": [0.0, 0.0, 0.0],
        }
    )

    assert result.requires_review is True
    assert result.max_stage1_score == 1.0
    assert {member["member_id"] for member in result.flagged_members} == {
        "GK00527181",
        "GK00140513",
        "GK00527185",
    }
    assert len(result.partnerships) == 1
    assert set(result.partnerships[0]["member_ids"]) == {"GK00527181", "GK00140513", "GK00527185"}
    assert result.partnerships[0]["union_coverage"] == 36 / 38
    assert result.partnerships[0]["is_section_b"] is True


def test_candidate_section_team_pattern_flags_all_roulette_cheating_members() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [0, 0])
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in STAGE2_FEATURE_COLUMNS}), [0, 0])
    sections = [
        (["GK00527181", "GK00140513", "GK00527185"], "CCS040502", list(range(14, 26)), 60000.0, 0.0),
        (["GK00537590", "GK00141731", "GK00537586"], "CCS041720", list(range(2, 14)), 60000.0, 180000.0),
        (["GK00301706", "GK00139170", "GK00301710"], "CCS039159", list(range(26, 38)), 60000.0, 0.0),
        (["GK00527633", "GK00527634", "GK00140526"], "CCS040515", [0, 1], 10000.0, 0.0),
    ]
    member_ids: list[str] = []
    ccs_ids: list[str] = []
    coverage: list[list[int]] = []
    amounts: list[list[float]] = []
    stakes: list[float] = []
    wins: list[float] = []
    for members, ccs_id, positions, stake, win_points in sections:
        mask = [1 if idx in positions else 0 for idx in range(38)]
        amount = stake / len(positions)
        for member in members:
            member_ids.append(member)
            ccs_ids.append(ccs_id)
            coverage.append(mask)
            amounts.append([amount if value else 0.0 for value in mask])
            stakes.append(stake)
            wins.append(win_points)

    result = DrawScorer(
        {
            "model_version": "partnership_v1",
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
            "use_candidate_store": True,
            "stage1_flag_threshold": 0.7,
        },
        source_run_id="run_test",
    ).score_candidate_draw(
        {
            "draw_id": 7305365,
            "qualifying_player_count": len(member_ids),
            "member_ids": member_ids,
            "ccs_ids": ccs_ids,
            "trans_date_min": pd.Timestamp("2026-05-08T11:56:36Z"),
            "trans_date_max": pd.Timestamp("2026-05-08T11:57:25Z"),
            "coverage_bytes": [bytes(row) for row in coverage],
            "amount_vector": amounts,
            "total_bet_amounts": stakes,
            "win_points": wins,
        }
    )

    assert result.requires_review is True
    assert result.max_stage1_score == 1.0
    assert {member["member_id"] for member in result.flagged_members} == set(member_ids)
    assert len(result.partnerships) == 1
    assert set(result.partnerships[0]["member_ids"]) == set(member_ids)
    assert result.partnerships[0]["union_coverage"] == 1.0
    assert result.partnerships[0]["is_section_b"] is True


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


def test_serving_ccs_context_is_neutral_when_promoted_lookup_is_empty() -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [1, 1])
    stage2_columns = list(dict.fromkeys(STAGE2_FEATURE_COLUMNS + CCS_FEATURE_COLUMNS))
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in stage2_columns}), [1, 1])
    scorer = DrawScorer(
        {
            "model_version": "partnership_v1",
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            "stage2_feature_columns": stage2_columns,
            "use_candidate_store": True,
            "ccs_features": {"enabled": True, "windows_days": [1, 7], "concentration_threshold": 0.7},
        },
        ccs_concentration_table=pd.DataFrame(),
    )
    stage1_scores = pd.DataFrame(
        {
            "member_id": ["A"],
            "ccs_id": ["C1"],
            "draw_id": [1],
            "draw_date": [pd.Timestamp("2026-05-07T00:00:00Z")],
            "best_partner_member_id": ["B"],
            "stage1_score": [0.8],
        }
    )

    result = scorer._attach_serving_ccs_context(stage1_scores)

    assert all(column in result.columns for column in CCS_FEATURE_COLUMNS)
    assert result[CCS_FEATURE_COLUMNS].sum().sum() == 0.0


def test_serving_ccs_context_prepares_promoted_lookup_once(monkeypatch) -> None:
    stage1_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage2_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
    stage1_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in PAIR_FEATURE_COLUMNS}), [1, 1])
    stage2_columns = list(dict.fromkeys(STAGE2_FEATURE_COLUMNS + CCS_FEATURE_COLUMNS))
    stage2_model.fit(pd.DataFrame({col: [0.0, 1.0] for col in stage2_columns}), [1, 1])

    calls = {"count": 0}
    original_prepare = draw_scorer_module.prepare_profit_frame

    def counting_prepare(profit_frame: pd.DataFrame) -> pd.DataFrame:
        calls["count"] += 1
        return original_prepare(profit_frame)

    monkeypatch.setattr(draw_scorer_module, "prepare_profit_frame", counting_prepare)

    scorer = DrawScorer(
        {
            "model_version": "partnership_v1",
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            "stage2_feature_columns": stage2_columns,
            "use_candidate_store": True,
            "ccs_features": {"enabled": True, "windows_days": [1, 7], "concentration_threshold": 0.7},
        },
        ccs_concentration_table=pd.DataFrame(
            [
                {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07"), "daily_profit": 80.0},
                {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07"), "daily_profit": 20.0},
            ]
        ),
    )
    stage1_scores = pd.DataFrame(
        {
            "member_id": ["A"],
            "ccs_id": ["C1"],
            "draw_id": [1],
            "draw_date": [pd.Timestamp("2026-05-07T00:00:00Z")],
            "best_partner_member_id": ["B"],
            "stage1_score": [0.8],
        }
    )

    first = scorer._attach_serving_ccs_context(stage1_scores)
    second = scorer._attach_serving_ccs_context(stage1_scores)

    assert calls["count"] == 1
    assert first[CCS_FEATURE_COLUMNS].equals(second[CCS_FEATURE_COLUMNS])
