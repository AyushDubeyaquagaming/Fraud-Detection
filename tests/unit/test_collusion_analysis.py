from __future__ import annotations

import pandas as pd

from fraud_detection.analysis.collusion import (
    bet_positions,
    compute_draw_cohort_metrics,
    decision_verdict,
    jaccard,
)
from scripts.inspect_bet_schema import inspect_docs


def test_bet_positions_keeps_positive_roulette_positions_only():
    bets = [
        {"number": "1", "bet_amount": 10},
        {"number": "01", "bet_amount": 5},
        {"number": "00", "bet_amount": 3},
        {"number": "9", "bet_amount": 0},
    ]
    assert bet_positions(bets) == {"1", "00"}


def test_compute_draw_cohort_metrics_detects_disjoint_coverage():
    rows = pd.DataFrame(
        [
            {"member_id": "A", "bets": '[{"number": "1", "bet_amount": 1}, {"number": "2", "bet_amount": 1}]', "total_bet_amount": 2},
            {"member_id": "B", "bets": '[{"number": "3", "bet_amount": 1}, {"number": "4", "bet_amount": 1}]', "total_bet_amount": 2},
            {"member_id": "C", "bets": '[{"number": "2", "bet_amount": 1}, {"number": "3", "bet_amount": 1}]', "total_bet_amount": 2},
        ]
    )
    metrics = compute_draw_cohort_metrics(rows)

    assert metrics["cohort_size"] == 3
    assert metrics["union_position_count"] == 4
    assert metrics["union_position_coverage"] == 4 / 38
    assert round(metrics["mean_pairwise_jaccard"], 6) == round((0 + 1 / 3 + 1 / 3) / 3, 6)


def test_decision_verdict_proceeds_for_high_coverage_low_overlap_signal():
    fraud = pd.DataFrame(
        {
            "cohort_size": [2, 2, 1],
            "union_position_coverage": [0.8, 0.7, 0.2],
            "mean_pairwise_jaccard": [0.05, 0.10, 0.0],
        }
    )
    baseline = pd.DataFrame(
        {
            "union_position_coverage": [0.3, 0.4],
            "mean_pairwise_jaccard": [0.3, 0.4],
        }
    )

    result = decision_verdict(fraud, baseline)
    assert result["verdict"] == "PROCEED"


def test_inspect_bet_schema_reports_number_only_verdict():
    docs = [
        {"bets": [{"number": "1", "bet_amount": 10}, {"number": "2", "bet_amount": 5}]},
        {"bets": '[{"number": "3", "bet_amount": 1}]'},
    ]
    report = inspect_docs(docs)

    assert report["verdict"] == "only bet.number present"
    assert sorted(report["fields"]) == ["bet_amount", "number"]
