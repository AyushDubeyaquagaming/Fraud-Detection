"""Unit tests for the Monitoring component."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.components.monitoring import Monitoring
from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureEngineeringArtifact,
    ModelEvaluationArtifact,
)
from fraud_detection.entity.config_entity import MonitoringConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def monitoring_config():
    return MonitoringConfig(
        enabled=True,
        reports_dir="monitoring",
        sample_size=200,
        monitored_features=["template_reuse_ratio", "avg_entropy", "draws_played"],
        drift_threshold=0.3,
        reference_from_current_metadata=True,
    )


def _make_raw_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "member_id": [f"m{i}" for i in range(n)],
        "total_bet_amount": rng.uniform(1, 100, n),
        "win_points": rng.uniform(0, 50, n),
        "bets": rng.integers(1, 20, n).astype(float),
    })


def _make_feature_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "member_id": [f"m{i}" for i in range(n)],
        "template_reuse_ratio": rng.uniform(0, 1, n),
        "avg_entropy": rng.uniform(0, 3, n),
        "draws_played": rng.integers(1, 50, n).astype(float),
        "event_fraud_flag": (rng.random(n) < 0.05).astype(int),
    })


def _make_scored_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "member_id": [f"m{i}" for i in range(n)],
        "hybrid_score": rng.uniform(0, 1, n),
    })


def _write_run_artifacts(run_dir: Path, raw_df, feat_df, scored_df) -> dict:
    (run_dir / "data_ingestion").mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_engineering").mkdir(parents=True, exist_ok=True)
    (run_dir / "model_evaluation").mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "data_ingestion" / "raw_data.parquet"
    feat_path = run_dir / "feature_engineering" / "player_features.parquet"
    scored_path = run_dir / "model_evaluation" / "scored_players.parquet"

    raw_df.to_parquet(raw_path, index=False)
    feat_df.to_parquet(feat_path, index=False)
    scored_df.to_parquet(scored_path, index=False)

    return {"raw": raw_path, "feat": feat_path, "scored": scored_path}


def _make_artifacts(paths: dict) -> tuple:
    ingestion = DataIngestionArtifact(
        raw_data_path=paths["raw"],
        ingestion_report_path=paths["raw"].parent / "report.json",
        row_count=300,
        member_count=300,
        source_type="parquet",
    )
    fe = FeatureEngineeringArtifact(
        player_features_path=paths["feat"],
        history_df_path=paths["feat"].parent / "history.parquet",
        fraud_player_count=5,
        dropped_positive_count=0,
        feature_columns=["template_reuse_ratio", "avg_entropy", "draws_played"],
        feature_summary_path=paths["feat"].parent / "summary.json",
        mode="training_eval",
    )
    ev = ModelEvaluationArtifact(
        scored_players_path=paths["scored"],
        capture_rate_table_path=paths["scored"].parent / "table.csv",
        evaluation_report_path=paths["scored"].parent / "report.json",
        gate_passed=True,
        combined_oos_capture_rate_top_5pct=0.55,
        combined_oos_lift_top_5pct=6.0,
        combined_oos_top_20pct=10,
    )
    return ingestion, fe, ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMonitoringSkipWhenNoPromotionMetadata:
    def test_skips_gracefully(self, monitoring_config, tmp_path):
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        ingestion, fe, ev = _make_artifacts(paths)

        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is False
        assert artifact.reports_dir is None


class TestMonitoringSkipWhenReferenceArtifactsMissing:
    def test_skips_when_run_dir_missing(self, monitoring_config, tmp_path):
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        # Write promotion_metadata pointing at a non-existent run_dir
        meta = {"gate_passed": True, "run_dir": str(tmp_path / "run_ghost")}
        (current_dir / "promotion_metadata.json").write_text(json.dumps(meta))

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        ingestion, fe, ev = _make_artifacts(paths)

        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is False

    def test_skips_when_metadata_missing_run_dir_key(self, monitoring_config, tmp_path):
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        meta = {"gate_passed": True}  # no run_dir key
        (current_dir / "promotion_metadata.json").write_text(json.dumps(meta))

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        ingestion, fe, ev = _make_artifacts(paths)

        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is False


class TestMonitoringGeneratesReports:
    def test_generates_reports_with_valid_reference(self, monitoring_config, tmp_path):
        ref_run_dir = tmp_path / "ref_run"
        cur_run_dir = tmp_path / "cur_run"
        current_dir = tmp_path / "current"
        current_dir.mkdir()

        ref_paths = _write_run_artifacts(ref_run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        cur_paths = _write_run_artifacts(cur_run_dir, _make_raw_df(300), _make_feature_df(200), _make_scored_df(200))

        meta = {"gate_passed": True, "run_dir": str(ref_run_dir)}
        (current_dir / "promotion_metadata.json").write_text(json.dumps(meta))

        ingestion, fe, ev = _make_artifacts(cur_paths)

        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, cur_run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is True
        assert artifact.reports_dir is not None
        assert artifact.drift_summary_path is not None
        assert artifact.drift_summary_path.exists()

        summary = json.loads(artifact.drift_summary_path.read_text())
        assert "overall_drift_detected" in summary
        assert "reports" in summary
        assert len(summary["reports"]) > 0


class TestMonitoringNonFatal:
    def test_corrupt_reference_does_not_raise(self, monitoring_config, tmp_path):
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        # Write corrupt JSON
        (current_dir / "promotion_metadata.json").write_text("not valid json{{{{")

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        ingestion, fe, ev = _make_artifacts(paths)

        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()  # must not raise

        assert artifact.monitoring_completed is False

    def test_disabled_returns_empty(self, tmp_path):
        config = MonitoringConfig(
            enabled=False,
            reports_dir="monitoring",
            sample_size=200,
            monitored_features=[],
            drift_threshold=0.3,
            reference_from_current_metadata=True,
        )
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        ingestion, fe, ev = _make_artifacts(paths)

        mon = Monitoring(config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is False


class TestSampling:
    def test_positive_heavy_sample_does_not_request_negative_rows(self):
        df = pd.DataFrame({
            "member_id": [f"m{i}" for i in range(6)],
            "event_fraud_flag": [1, 1, 1, 1, 1, 0],
        })

        sampled = Monitoring.__module__  # keep import path stable for local test discovery
        sampled = __import__(sampled, fromlist=["_sample"])._sample(df, 3, "event_fraud_flag")

        assert len(sampled) == 3
        assert sampled["event_fraud_flag"].eq(1).all()


class TestSelfReferenceGuard:
    def test_skips_when_reference_run_dir_equals_current_run_dir(self, monitoring_config, tmp_path):
        """Force-promote ordering bug regression test: if promotion_metadata.json
        points at the same run we're now monitoring, monitoring must skip
        rather than loading the same large parquet twice and OOMing.
        """
        run_dir = tmp_path / "run_self"
        run_dir.mkdir()
        current_dir = tmp_path / "current"
        current_dir.mkdir()

        paths = _write_run_artifacts(run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())

        # promotion_metadata points at the SAME run we are monitoring
        meta = {"gate_passed": True, "run_dir": str(run_dir)}
        (current_dir / "promotion_metadata.json").write_text(json.dumps(meta))

        ingestion, fe, ev = _make_artifacts(paths)
        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, run_dir)
        artifact = mon.initiate_monitoring()

        assert artifact.monitoring_completed is False
        # No reports written because we skipped before any drift work
        assert artifact.reports_dir is None


class TestBoundedParquetSampling:
    def test_does_not_call_pd_read_parquet_on_full_file(self, monitoring_config, tmp_path, monkeypatch):
        """Regression test for the RAM blowup: sampling must use pyarrow row
        groups, NOT pd.read_parquet on the full file. We replace pd.read_parquet
        with a sentinel that fails the test if invoked on a path that wasn't
        already written by the test fixture (i.e. used as the legitimate
        small-file fast path)."""
        ref_run_dir = tmp_path / "ref_run"
        cur_run_dir = tmp_path / "cur_run"
        current_dir = tmp_path / "current"
        current_dir.mkdir()

        ref_paths = _write_run_artifacts(ref_run_dir, _make_raw_df(), _make_feature_df(), _make_scored_df())
        cur_paths = _write_run_artifacts(cur_run_dir, _make_raw_df(300), _make_feature_df(200), _make_scored_df(200))

        meta = {"gate_passed": True, "run_dir": str(ref_run_dir)}
        (current_dir / "promotion_metadata.json").write_text(json.dumps(meta))

        # Track all calls to pd.read_parquet
        from fraud_detection.components import monitoring as mon_module
        calls = []
        original = mon_module.pd.read_parquet

        def tracked(path, *args, **kwargs):
            calls.append(str(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(mon_module.pd, "read_parquet", tracked)

        ingestion, fe, ev = _make_artifacts(cur_paths)
        mon = Monitoring(monitoring_config, current_dir, ingestion, fe, ev, cur_run_dir)
        mon.initiate_monitoring()

        # The bounded sampler uses pq.ParquetFile.read() / read_row_group(),
        # not pd.read_parquet, regardless of file size. So calls must be empty.
        assert calls == [], (
            f"Monitoring must NOT call pd.read_parquet — it loads the whole "
            f"file into RAM. Got {len(calls)} call(s): {calls}"
        )

    def test_bounded_sample_returns_n_rows_for_large_file(self, tmp_path):
        """The bounded sampler must return exactly n rows when the input has
        more than n rows, regardless of file size or row-group layout."""
        from fraud_detection.components.monitoring import _sample_parquet_bounded

        rng = np.random.default_rng(0)
        big = pd.DataFrame({
            "x": rng.uniform(0, 1, 10_000),
            "y": rng.uniform(0, 1, 10_000),
        })
        path = tmp_path / "big.parquet"
        big.to_parquet(path, index=False, row_group_size=500)  # 20 row groups

        sampled = _sample_parquet_bounded(path, n=300)
        assert sampled is not None
        assert len(sampled) == 300

    def test_bounded_sample_returns_full_file_when_smaller_than_n(self, tmp_path):
        from fraud_detection.components.monitoring import _sample_parquet_bounded

        small = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = tmp_path / "small.parquet"
        small.to_parquet(path, index=False)

        sampled = _sample_parquet_bounded(path, n=100)
        assert sampled is not None
        assert len(sampled) == 3

    def test_bounded_sample_returns_none_for_missing_file(self, tmp_path):
        from fraud_detection.components.monitoring import _sample_parquet_bounded

        sampled = _sample_parquet_bounded(tmp_path / "nope.parquet", n=100)
        assert sampled is None
