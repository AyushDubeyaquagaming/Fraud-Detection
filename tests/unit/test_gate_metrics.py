from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.components.model_evaluation import _capture_stats


def test_stage2_capture_stats_for_perfect_top_rank():
    scores = pd.Series([0.99, 0.98, 0.10, 0.05])
    labels = pd.Series([1, 1, 0, 0])

    stats = _capture_stats(scores, labels, k=2)

    assert stats["k"] == 2
    assert stats["captured_fraud"] == 2
    assert stats["capture_rate"] == 1.0
    assert stats["precision"] == 1.0
    assert stats["lift"] == pytest.approx(2.0)


def test_stage2_capture_stats_handles_no_positive_labels():
    scores = pd.Series([0.99, 0.50, 0.10])
    labels = pd.Series([0, 0, 0])

    stats = _capture_stats(scores, labels, k=2)

    assert stats["captured_fraud"] == 0
    assert stats["capture_rate"] == 0.0
    assert stats["lift"] == 0.0
