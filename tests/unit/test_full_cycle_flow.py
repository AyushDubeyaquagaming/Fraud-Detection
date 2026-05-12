from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from orchestration.flows import full_cycle_flow
from fraud_detection.utils.common import read_json, write_json


@dataclass
class _CandidateSummary:
    chunks_total: int = 2
    succeeded: int = 1
    skipped: int = 1
    failed: int = 0
    total_rows: int = 12
    elapsed_seconds: float = 1.5

    @property
    def exit_code(self) -> int:
        return 0


@dataclass
class _CcsSummary:
    status: str = "FINISHED"
    days_processed: int = 2
    days_succeeded: int = 1
    skipped_days: int = 1
    failed_days: int = 0
    rows_written: int = 5
    elapsed_seconds: float = 0.5

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "days_processed": self.days_processed,
            "days_succeeded": self.days_succeeded,
            "skipped_days": self.skipped_days,
            "failed_days": self.failed_days,
            "rows_written": self.rows_written,
            "elapsed_seconds": self.elapsed_seconds,
        }


class _FakeMlflow:
    def __init__(self):
        self.ended_status = None

    def set_tag(self, *_args, **_kwargs):
        return None

    def active_run(self):
        return object() if self.ended_status is None else None

    def end_run(self, status=None):
        self.ended_status = status


def test_full_cycle_runs_batch_only_after_promotion(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    ccs_config_path = tmp_path / "ccs.yaml"
    batch_config_path = tmp_path / "batch.yaml"
    candidate_config_path = tmp_path / "candidate.yaml"
    config = {
        "pipeline": {
            "artifact_root": str(tmp_path / "artifacts"),
            "current_dir": str(tmp_path / "current"),
            "weekly_serving_snapshot_config": str(batch_config_path),
        },
        "partnership": {
            "use_candidate_store": True,
            "candidate_window": {"start_date": "2026-01-01", "end_date": "2026-01-08"},
            "ccs_features": {"enabled": True, "windows_days": [1, 7]},
        },
        "mlflow": {"experiment_name": "test"},
    }
    ccs_config = {
        "output": {"base_path": str(tmp_path / "ccs_profit"), "parquet_compression": "zstd"},
        "mongo": {},
        "extraction": {"timestamp_field": "trans_date"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ccs_config_path.write_text(yaml.safe_dump(ccs_config), encoding="utf-8")
    batch_config_path.write_text("{}", encoding="utf-8")
    candidate_config_path.write_text("{}", encoding="utf-8")

    class FakeExtractor:
        def __init__(self, _config):
            pass

        def run(self, *, start_date, end_date, force=False):
            assert force is False
            assert start_date.date().isoformat() == "2026-05-04"
            assert end_date.date().isoformat() == "2026-05-11"
            return _CandidateSummary()

    class FakeTrainingPipeline:
        def __init__(self, *, config_path, manage_mlflow, run_batch_scoring_on_promotion, mlflow_source):
            assert Path(config_path).name == "resolved_training_config.yaml"
            resolved = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            assert resolved["partnership"]["candidate_window"] == {
                "start_date": "2026-05-04",
                "end_date": "2026-05-11",
            }
            assert manage_mlflow is False
            assert run_batch_scoring_on_promotion is False
            assert mlflow_source == "full_cycle"

        def run(self):
            run_dir = tmp_path / "artifacts" / "runs" / "run_test"
            write_json({"run_id": "run_test", "status": "FINISHED", "promoted": True}, run_dir / "run_metadata.json")
            return run_dir

    class FakeBatchScoringPipeline:
        def __init__(self, *, config_path):
            assert Path(config_path) == batch_config_path

        def run(self):
            output_dir = tmp_path / "artifacts" / "current" / "batch_scoring"
            write_json({"draws_scored": 3}, output_dir / "batch_scoring_report.json")
            return output_dir

    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(full_cycle_flow, "_start_mlflow_run", lambda *_args, **_kwargs: fake_mlflow)
    monkeypatch.setattr(full_cycle_flow, "log_artifact_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(full_cycle_flow, "log_metrics_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(full_cycle_flow, "log_params_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(full_cycle_flow, "load_candidate_extraction_config", lambda _path: {})
    monkeypatch.setattr(full_cycle_flow, "CandidateDrawExtractor", FakeExtractor)
    ccs_call = {}

    def fake_build_ccs_daily_profit(start_date, end_date, *_args, **_kwargs):
        ccs_call["start_date"] = start_date
        ccs_call["end_date"] = end_date
        return _CcsSummary()

    monkeypatch.setattr(full_cycle_flow, "build_ccs_daily_profit", fake_build_ccs_daily_profit)
    monkeypatch.setattr(full_cycle_flow, "TrainingPipeline", FakeTrainingPipeline)
    monkeypatch.setattr(full_cycle_flow, "BatchScoringPipeline", FakeBatchScoringPipeline)

    result = full_cycle_flow.run_full_cycle(
        config_path=config_path,
        candidate_config_path=candidate_config_path,
        ccs_config_path=ccs_config_path,
        batch_config_path=batch_config_path,
        start_date="2026-05-04",
        end_date="2026-05-11",
    )

    assert result["status"] == "FINISHED"
    assert result["candidate_extraction"]["total_rows"] == 12
    assert result["ccs_profit"]["rows_written"] == 5
    assert result["ccs_profit_start_date"] == "2026-04-28"
    assert result["ccs_profit_end_date"] == "2026-05-10"
    assert Path(result["training_config_path"]).name == "resolved_training_config.yaml"
    assert ccs_call["start_date"].isoformat() == "2026-04-28"
    assert ccs_call["end_date"].isoformat() == "2026-05-10"
    assert result["training"]["promoted"] is True
    assert result["batch_scoring"]["draws_scored"] == 3
    assert fake_mlflow.ended_status == "FINISHED"
    summary = read_json(Path(result["full_cycle_dir"]) / "full_cycle_summary.json")
    assert summary["status"] == "FINISHED"
