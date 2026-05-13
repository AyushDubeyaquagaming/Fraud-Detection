from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow

from fraud_detection.components.model_pusher import ModelPusher
from fraud_detection.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelPusherConfig
from fraud_detection.utils import mlflow_utils


def test_partnership_pusher_writes_manifest_and_bundle(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_test"
    train_dir = run_dir / "model_training"
    eval_dir = run_dir / "model_evaluation"
    current_dir = tmp_path / "current"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    current_dir.mkdir()

    bundle_path = train_dir / "model_bundle.joblib"
    stage1_path = train_dir / "stage1_model.joblib"
    stage2_path = train_dir / "stage2_model.joblib"
    report_path = train_dir / "training_report.json"
    joblib.dump({"model_version": "partnership_v1"}, bundle_path)
    joblib.dump({"stage": 1}, stage1_path)
    joblib.dump({"stage": 2}, stage2_path)
    report_path.write_text(json.dumps({"stage2_alert_threshold": 0.65}))

    scored_path = eval_dir / "stage2_holdout_predictions.parquet"
    eval_path = eval_dir / "evaluation_report.json"
    import pandas as pd

    pd.DataFrame({"member_id": ["A"], "stage2_score": [0.9]}).to_parquet(scored_path, index=False)
    eval_path.write_text(json.dumps({"gate_passed": True}))

    train_artifact = ModelTrainingArtifact(
        training_report_path=report_path,
        feature_columns=[],
        stage1_model_path=stage1_path,
        stage2_model_path=stage2_path,
        model_bundle_path=bundle_path,
    )
    eval_artifact = ModelEvaluationArtifact(
        stage2_holdout_predictions_path=scored_path,
        capture_rate_table_path=eval_dir / "capture.csv",
        evaluation_report_path=eval_path,
        gate_passed=True,
        stage2_capture_rate_top_5pct=0.5,
        stage2_lift_top_5pct=4.0,
        stage2_top_50_captured=0,
    )
    config = ModelPusherConfig(
        current_dir=current_dir,
        model_version="partnership_v1",
        registered_model_name="fraud_detection_partnership_v1",
        register_on_promotion=False,
    )

    artifact = ModelPusher(config, train_artifact, eval_artifact).initiate_model_pusher()

    assert artifact.promoted is True
    assert (current_dir / "model_bundle.joblib").exists()
    manifest = json.loads((current_dir / "serving_manifest.json").read_text())
    assert manifest["model_version"] == "partnership_v1"
    assert manifest["model_bundle_file"] == "model_bundle.joblib"


def test_log_lineage_bundle_model_returns_model_directory_uri(monkeypatch, tmp_path: Path):
    bundle_path = tmp_path / "model_bundle.joblib"
    bundle_path.write_text("bundle", encoding="utf-8")
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.as_uri())

    with mlflow.start_run() as run:
        uri = mlflow_utils.log_lineage_bundle_model(bundle_path)

    assert uri == f"runs:/{run.info.run_id}/model_bundle"
