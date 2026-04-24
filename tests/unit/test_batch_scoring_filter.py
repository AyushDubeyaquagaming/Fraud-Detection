"""Operational filter unit tests for batch scoring.

Verifies that the Phase 2 operational_filter (applied at SCORING time only)
correctly bounds the alert-queue source population and does not affect the
full scored parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.pipeline.batch_scoring_pipeline import (
    _load_parquet_to_window,
    _stage_parquet_window_to_path,
)


def _apply_operational_filter(
    player_df: pd.DataFrame,
    *,
    enabled: bool,
    min_draws_played: int,
) -> pd.DataFrame:
    """Mirror of the filter block in BatchScoringPipeline.

    Lifted into a standalone helper so this unit test can exercise the
    semantics without instantiating the full pipeline.
    """
    if enabled and "draws_played" in player_df.columns:
        return player_df.loc[player_df["draws_played"] >= min_draws_played]
    return player_df


def _make_synthetic_scored(n_heavy: int = 100, n_light: int = 200) -> pd.DataFrame:
    heavy = pd.DataFrame({
        "member_id": [f"H{i:04d}" for i in range(n_heavy)],
        "draws_played": [50 + i for i in range(n_heavy)],
        "risk_score": [0.9 - i * 0.001 for i in range(n_heavy)],
    })
    light = pd.DataFrame({
        "member_id": [f"L{i:04d}" for i in range(n_light)],
        "draws_played": [i % 10 for i in range(n_light)],   # 0..9
        "risk_score": [0.95 + i * 0.0001 for i in range(n_light)],  # artificially higher
    })
    return pd.concat([heavy, light], ignore_index=True)


def test_filter_disabled_keeps_all_rows():
    df = _make_synthetic_scored()
    out = _apply_operational_filter(df, enabled=False, min_draws_played=10)
    assert len(out) == len(df)


def test_filter_excludes_low_activity_players():
    df = _make_synthetic_scored(n_heavy=100, n_light=200)
    out = _apply_operational_filter(df, enabled=True, min_draws_played=10)
    # All 200 light players have draws_played in [0, 9], all excluded.
    # All 100 heavy players have draws_played in [50, 149], all kept.
    assert len(out) == 100
    assert (out["draws_played"] >= 10).all()


def test_filter_alert_queue_size_is_bounded():
    """Even though light players have artificially higher risk_score, the
    filter must remove them so the alert queue reflects active-player ranking."""
    df = _make_synthetic_scored(n_heavy=100, n_light=200)
    filtered = _apply_operational_filter(df, enabled=True, min_draws_played=10)
    alert_queue = filtered.sort_values("risk_score", ascending=False).head(50)
    # Every alert is a heavy player
    assert alert_queue["member_id"].str.startswith("H").all()
    assert len(alert_queue) == 50


def test_filter_handles_missing_draws_played_column():
    """If draws_played is absent (edge case), the filter should no-op."""
    df = pd.DataFrame({"member_id": ["A", "B"], "risk_score": [0.5, 0.9]})
    out = _apply_operational_filter(df, enabled=True, min_draws_played=10)
    assert len(out) == 2


def test_parquet_window_loader_reads_only_requested_window(tmp_path: Path):
    df = pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "trans_date": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-05T00:00:00Z",
                    "2026-04-10T00:00:00Z",
                ],
                utc=True,
            ),
            "draw_id": [1, 2, 3],
        }
    )
    parquet_path = tmp_path / "weekly_slice.parquet"
    df.to_parquet(parquet_path, index=False)

    filtered_df, metadata = _load_parquet_to_window(
        parquet_path,
        {
            "timestamp_field": "trans_date",
            "start_date": "2026-04-02T00:00:00+00:00",
            "end_date": "2026-04-09T00:00:00+00:00",
        },
    )

    assert filtered_df["member_id"].tolist() == ["B"]
    assert metadata["timestamp_field"] == "trans_date"
    assert metadata["window_start"] == "2026-04-02T00:00:00+00:00"
    assert metadata["window_end"] == "2026-04-09T00:00:00+00:00"


def test_parquet_window_loader_handles_timezone_naive_parquet(tmp_path: Path):
    df = pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "trans_date": pd.to_datetime(
                [
                    "2026-04-01 00:00:00",
                    "2026-04-05 00:00:00",
                    "2026-04-10 00:00:00",
                ]
            ),
            "draw_id": [1, 2, 3],
        }
    )
    parquet_path = tmp_path / "weekly_slice_naive.parquet"
    df.to_parquet(parquet_path, index=False)

    filtered_df, metadata = _load_parquet_to_window(
        parquet_path,
        {
            "timestamp_field": "trans_date",
            "start_date": "2026-04-02T00:00:00+00:00",
            "end_date": "2026-04-09T00:00:00+00:00",
        },
    )

    assert filtered_df["member_id"].tolist() == ["B"]
    assert metadata["window_start"] == "2026-04-02T00:00:00+00:00"
    assert metadata["window_end"] == "2026-04-09T00:00:00+00:00"


def test_parquet_window_loader_uses_bounded_pyarrow_fallback(tmp_path: Path, monkeypatch):
    df = pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "trans_date": pd.to_datetime(
                [
                    "2026-04-01 00:00:00",
                    "2026-04-05 00:00:00",
                    "2026-04-10 00:00:00",
                ]
            ),
            "draw_id": [1, 2, 3],
        }
    )
    parquet_path = tmp_path / "weekly_slice_pyarrow_fallback.parquet"
    df.to_parquet(parquet_path, index=False)

    def fake_read_parquet(*args, **kwargs):
        if kwargs.get("filters"):
            raise TypeError("simulated pandas parquet filter mismatch")
        raise AssertionError("_load_parquet_to_window should not fall back to an unfiltered full read")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    filtered_df, metadata = _load_parquet_to_window(
        parquet_path,
        {
            "timestamp_field": "trans_date",
            "start_date": "2026-04-02T00:00:00+00:00",
            "end_date": "2026-04-09T00:00:00+00:00",
        },
    )

    assert filtered_df["member_id"].tolist() == ["B"]
    assert metadata["window_start"] == "2026-04-02T00:00:00+00:00"
    assert metadata["window_end"] == "2026-04-09T00:00:00+00:00"


def test_stage_parquet_window_to_path_writes_only_requested_window(tmp_path: Path):
    df = pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "trans_date": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-05T00:00:00Z",
                    "2026-04-10T00:00:00Z",
                ],
                utc=True,
            ),
            "draw_id": [1, 2, 3],
        }
    )
    source_path = tmp_path / "weekly_source.parquet"
    staged_path = tmp_path / "staged.parquet"
    df.to_parquet(source_path, index=False)

    stats, metadata = _stage_parquet_window_to_path(
        source_path,
        staged_path,
        {
            "timestamp_field": "trans_date",
            "start_date": "2026-04-02T00:00:00+00:00",
            "end_date": "2026-04-09T00:00:00+00:00",
        },
    )

    staged_df = pd.read_parquet(staged_path)

    assert staged_df["member_id"].tolist() == ["B"]
    assert stats["row_count"] == 1
    assert stats["member_count"] == 1
    assert metadata["window_start"] == "2026-04-02T00:00:00+00:00"
    assert metadata["window_end"] == "2026-04-09T00:00:00+00:00"
