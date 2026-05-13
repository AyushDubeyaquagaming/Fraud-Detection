from __future__ import annotations

from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS
from fraud_detection.components.partnership_features import STAGE2_FEATURE_COLUMNS
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
import numpy as np


class _ConstantModel:
    classes_ = [0, 1]

    def __init__(self, score: float = 0.8):
        self.score = score

    def predict_proba(self, frame):
        return np.array([[1.0 - self.score, self.score] for _ in range(len(frame))])


def _bundle():
    return {
        "model_version": "partnership_v1",
        "use_candidate_store": True,
        "stage1_model": _ConstantModel(0.8),
        "stage2_model": _ConstantModel(0.2),
        "stage1_feature_columns": PAIR_FEATURE_COLUMNS,
        "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
        "pair_rules": {
            "strict_min_ratio_similarity": 0.80,
            "nearmiss_min_union": 36,
            "nearmiss_max_overlap": 2,
            "nearmiss_min_ratio_similarity": 0.70,
            "min_total_bet_amount": 1000.0,
            "stage1_flag_threshold": 0.70,
        },
    }


def test_full_payload_like_draw_scores_with_pair_scan() -> None:
    left_bets = [{"number": str(number), "bet_amount": 10.0} for number in range(1, 20)]
    right_bets = [{"number": str(number), "bet_amount": 10.0} for number in range(20, 37)] + [
        {"number": "0", "bet_amount": 10.0},
        {"number": "00", "bet_amount": 10.0},
    ]
    rows = [
        {
            "draw_id": 1,
            "member_id": "A",
            "ccs_id": "CA",
            "total_bet_amount": 1000.0,
            "win_points": 1200.0,
            "bets": left_bets,
            "trans_date": "2026-04-27T00:00:00Z",
        },
        {
            "draw_id": 1,
            "member_id": "B",
            "ccs_id": "CB",
            "total_bet_amount": 1000.0,
            "win_points": 1000.0,
            "bets": right_bets,
            "trans_date": "2026-04-27T00:00:00Z",
        },
    ]

    result = DrawScorer(_bundle()).score_draw(__import__("pandas").DataFrame(rows))

    assert result.requires_review
    assert result.partnerships
    assert sorted(result.candidate_members) == ["A", "B"]


def test_draw_id_compatibility_path_can_still_score_raw_rows() -> None:
    test_full_payload_like_draw_scores_with_pair_scan()
