"""Unit tests for the Phase 2 rebaseline gate metrics.

Covers:
  - _capture_count_topk (fixed-count thresholds)
  - _compute_capture_stats (lift, precision, capture_rate)
  - Gate pass/fail semantics on the baseline archive numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.components.model_evaluation import (
    _capture_count_topk,
    _compute_capture_stats,
)


def _fixture(n: int = 1000, n_fraud: int = 10, perfect: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    if perfect:
        scores = np.concatenate([np.full(n_fraud, 100.0), rng.uniform(0, 1, n - n_fraud)])
    else:
        scores = rng.uniform(0, 1, n)
    return pd.Series(scores), pd.Series(labels)


def test_capture_count_topk_perfect_ranking():
    scores, labels = _fixture(n=100, n_fraud=5, perfect=True)
    assert _capture_count_topk(scores, labels, 5) == 5
    assert _capture_count_topk(scores, labels, 10) == 5


def test_capture_count_topk_clamps_to_length():
    scores, labels = _fixture(n=10, n_fraud=3, perfect=True)
    assert _capture_count_topk(scores, labels, 1000) == 3  # can't exceed total fraud


def test_compute_capture_stats_lift_and_capture_rate():
    """Perfect ranking: top 1% contains all 10 positives out of 1000."""
    scores, labels = _fixture(n=1000, n_fraud=10, perfect=True)
    stats = _compute_capture_stats(
        scores, labels, percentiles=[0.01, 0.05], fixed_counts=[50, 500]
    )
    # top_1pct → k=10, all 10 fraud captured
    assert stats["top_1pct"]["k"] == 10
    assert stats["top_1pct"]["captured_fraud"] == 10
    assert stats["top_1pct"]["capture_rate"] == 1.0
    # precision = 10/10 = 1.0; base rate = 10/1000 = 0.01; lift = 100x
    assert stats["top_1pct"]["precision"] == pytest.approx(1.0)
    assert stats["top_1pct"]["lift"] == pytest.approx(100.0)

    # top_50 → all 10 captured; precision=10/50=0.2; lift=20x
    assert stats["top_50"]["captured_fraud"] == 10
    assert stats["top_50"]["precision"] == pytest.approx(0.2)
    assert stats["top_50"]["lift"] == pytest.approx(20.0)


def test_compute_capture_stats_random_ranking_lift_near_1():
    scores, labels = _fixture(n=1000, n_fraud=50, perfect=False, seed=42)
    stats = _compute_capture_stats(scores, labels, percentiles=[0.10], fixed_counts=[100])
    # Random ranking: lift should be close to 1.0 on average. Allow a wide band.
    assert 0.3 < stats["top_10pct"]["lift"] < 3.0


def test_compute_capture_stats_zero_fraud():
    scores = pd.Series(np.random.rand(100))
    labels = pd.Series(np.zeros(100, dtype=int))
    stats = _compute_capture_stats(scores, labels, percentiles=[0.05], fixed_counts=[10])
    assert stats["top_5pct"]["captured_fraud"] == 0
    assert stats["top_5pct"]["capture_rate"] == 0.0
    assert stats["top_5pct"]["lift"] == 0.0  # base_rate=0, lift defined as 0


def test_gate_passes_on_baseline_archive_numbers():
    """Baseline archive (14 fraud in 1,045): OOS top 5% = 6/14, lift ~8.6x.

    This reproduces the computation without loading the archive — we just show
    that the chosen thresholds (min_capture_rate=0.40, min_lift=5.0) pass for
    a score distribution that mirrors the archive's behavior.
    """
    n = 1045
    n_fraud = 14
    # Construct scores so that the top 5% (52 players) contains exactly 6 fraud —
    # matching the archive's OOS capture.
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng = np.random.default_rng(123)
    scores = rng.uniform(0, 1, n)
    # Bump 6 fraud players into the top 52
    scores[:6] = 0.999 - rng.uniform(0, 0.0001, 6)
    # Leave the other 8 fraud in the noise
    scores[6:n_fraud] = rng.uniform(0, 0.5, n_fraud - 6)

    stats = _compute_capture_stats(
        pd.Series(scores), pd.Series(labels), percentiles=[0.05], fixed_counts=[]
    )
    cap5 = stats["top_5pct"]["capture_rate"]
    lift5 = stats["top_5pct"]["lift"]
    assert cap5 == pytest.approx(6 / 14, abs=0.01), f"capture_rate={cap5:.3f}"
    assert cap5 >= 0.40, "gate should pass on baseline-archive capture rate"
    assert lift5 >= 5.0, f"gate should pass on baseline-archive lift (was {lift5:.2f}x)"


def test_gate_fails_on_live_cohort_numbers():
    """Live cohort (10 fraud in 105,245): OOS top 5% = 2/10, lift ~4.0x.

    Both thresholds should fail.
    """
    n = 105_245
    n_fraud = 10
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng = np.random.default_rng(456)
    scores = rng.uniform(0, 1, n)
    # Put 2 fraud in the top 5% (~5,262 players), leave 8 in noise
    top_k = int(n * 0.05)
    scores[:2] = 1.0 - rng.uniform(0, 0.0001, 2)
    scores[2:n_fraud] = rng.uniform(0, 0.5, n_fraud - 2)

    stats = _compute_capture_stats(
        pd.Series(scores), pd.Series(labels), percentiles=[0.05], fixed_counts=[]
    )
    cap5 = stats["top_5pct"]["capture_rate"]
    lift5 = stats["top_5pct"]["lift"]
    assert cap5 == pytest.approx(2 / 10, abs=0.01)
    assert cap5 < 0.40, "live-cohort capture rate should fail the gate"
    assert lift5 < 5.0, "live-cohort lift should fail the gate"
