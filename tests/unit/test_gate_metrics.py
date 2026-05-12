from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.components.model_evaluation import ModelEvaluation, _capture_stats
from fraud_detection.entity.artifact_entity import ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.utils.common import read_json


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


def test_model_evaluation_promotes_when_labels_are_unavailable(tmp_path):
    train_dir = tmp_path / "model_training"
    train_dir.mkdir()
    stage1_path = train_dir / "stage1_oof_predictions.parquet"
    stage2_features_path = train_dir / "stage2_features.parquet"
    stage2_predictions_path = train_dir / "stage2_predictions.parquet"
    training_report_path = train_dir / "training_report.json"
    pd.DataFrame({"member_id": ["A"]}).to_parquet(stage1_path, index=False)
    pd.DataFrame({"member_id": ["A"], "label_gold_member": [0]}).to_parquet(stage2_features_path, index=False)
    pd.DataFrame({"member_id": ["A"], "stage2_score": [0.1]}).to_parquet(stage2_predictions_path, index=False)
    training_report_path.write_text('{"model_version":"partnership_v1"}')
    artifact = ModelTrainingArtifact(
        training_report_path=training_report_path,
        feature_columns=[],
        stage1_oof_predictions_path=stage1_path,
    )

    result = ModelEvaluation(ModelEvaluationConfig(output_dir=tmp_path / "evaluation"), artifact).initiate_model_evaluation()

    report = read_json(result.evaluation_report_path)
    assert result.gate_passed is True
    assert report["label_status"] == "unavailable"
    assert report["gate_reason"] == "no_analyst_labels_available_promote_latest_artifacts"
