"""Unit tests for the MLflow Model Registry integration in ModelPusher.

These tests do NOT touch a real MLflow server. They mock the MLflow client
surface so we can assert exactly which calls happen on a successful gate-pass
promotion, and confirm that registry failures never block promotion (registry
is best-effort; filesystem promotion is the source of truth for serving).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fraud_detection.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainingArtifact,
)
from fraud_detection.entity.config_entity import ModelPusherConfig


def _make_artifacts(tmp_path: Path) -> tuple[ModelTrainingArtifact, ModelEvaluationArtifact]:
    """Build minimal training + evaluation artifacts on disk for ModelPusher.

    Uses picklable placeholders (dicts) instead of MagicMock because
    ModelPusher round-trips these through joblib, which cannot serialize
    mock objects on Windows.
    """
    import joblib
    import numpy as np
    import pandas as pd

    run_dir = tmp_path / "runs" / "run_test"
    train_dir = run_dir / "model_training"
    eval_dir = run_dir / "model_evaluation"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    iso = {"_kind": "iso_forest_stub"}
    km = {"_kind": "kmeans_stub"}
    mahal = {"mean_vec": np.zeros(3), "cov_inv": np.eye(3)}
    scalers = {
        "scaler_unsup": {"_kind": "scaler_unsup_stub"},
        "scaler_operational": {"_kind": "scaler_operational_stub"},
    }
    lrs = {"lr_operational": {"_kind": "lr_operational_stub"}}

    iso_path = train_dir / "iso_forest.joblib"
    km_path = train_dir / "kmeans.joblib"
    mahal_path = train_dir / "mahal_stats.joblib"
    scaler_path = train_dir / "scalers.joblib"
    lr_path = train_dir / "lr.joblib"
    joblib.dump(iso, iso_path)
    joblib.dump(km, km_path)
    joblib.dump(mahal, mahal_path)
    joblib.dump(scalers, scaler_path)
    joblib.dump(lrs, lr_path)

    training_report = {
        "log1p_cols": [],
        "style_columns": [],
        "style_log1p_cols": [],
    }
    train_report_path = train_dir / "training_report.json"
    train_report_path.write_text(json.dumps(training_report))

    scored = pd.DataFrame(
        {
            "iso_forest_score": [0.1, 0.2, 0.3],
            "mahalanobis_dist": [1.0, 2.0, 3.0],
            "cluster_distance": [0.5, 1.0, 1.5],
            "primary_ccs_id": ["A", "B", "A"],
            "ccs_player_count": [10, 5, 10],
            "ccs_total_staked": [100.0, 50.0, 100.0],
            "ccs_avg_bet": [10.0, 10.0, 10.0],
        }
    )
    scored_path = eval_dir / "scored_players.parquet"
    scored.to_parquet(scored_path)

    eval_report = {
        "capture_rates": {"combined_oos": {"top_5pct": 0.5}},
        "capture_stats": {},
        "risk_p80": 0.8,
        "risk_p95": 0.95,
        "anomaly_weight": 0.6,
        "supervised_weight": 0.4,
        "anomaly_component_weights": {},
    }
    eval_report_path = eval_dir / "evaluation_report.json"
    eval_report_path.write_text(json.dumps(eval_report))

    training_artifact = ModelTrainingArtifact(
        iso_forest_path=iso_path,
        kmeans_path=km_path,
        mahalanobis_stats_path=mahal_path,
        scaler_path=scaler_path,
        lr_operational_path=lr_path,
        training_report_path=train_report_path,
        feature_columns=["f1", "f2", "f3"],
    )

    eval_artifact = ModelEvaluationArtifact(
        scored_players_path=scored_path,
        capture_rate_table_path=eval_dir / "capture_rate_table.csv",
        evaluation_report_path=eval_report_path,
        gate_passed=True,
        combined_oos_capture_rate_top_5pct=0.55,
        combined_oos_lift_top_5pct=11.0,
        combined_oos_top_20pct=42,
    )

    return training_artifact, eval_artifact


def _pusher_config(current_dir: Path, **overrides) -> ModelPusherConfig:
    base = dict(
        current_dir=current_dir,
        manifest_file="serving_manifest.json",
        model_version="hybrid_v1",
        min_capture_rate_top_5pct=0.40,
        min_lift_top_5pct=5.0,
        register_on_promotion=True,
        registered_model_name="fraud_detection_hybrid",
        archive_existing_staging=True,
        auto_promote_to_production=False,
    )
    base.update(overrides)
    return ModelPusherConfig(**base)


def test_register_model_to_staging_calls_mlflow_correctly():
    """Direct test of the helper: must register, transition to Staging, and
    archive existing staging versions."""
    from fraud_detection.utils.mlflow_utils import register_model_to_staging

    fake_version = MagicMock()
    fake_version.version = "7"
    fake_version.run_id = "run-xyz"

    with patch("mlflow.register_model", return_value=fake_version) as mock_register, patch(
        "mlflow.tracking.MlflowClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        result = register_model_to_staging(
            artifact_uri="runs:/run-xyz/model_bundle/model_bundle.joblib",
            registered_name="fraud_detection_hybrid",
            description="desc",
            tags={"git_sha": "abc"},
            archive_existing_staging=True,
        )

    assert result == {
        "name": "fraud_detection_hybrid",
        "version": "7",
        "stage": "Staging",
        "run_id": "run-xyz",
    }
    mock_register.assert_called_once()
    args, kwargs = mock_register.call_args
    assert kwargs["name"] == "fraud_detection_hybrid"
    assert kwargs["model_uri"] == "runs:/run-xyz/model_bundle/model_bundle.joblib"

    mock_client.transition_model_version_stage.assert_called_once_with(
        name="fraud_detection_hybrid",
        version="7",
        stage="Staging",
        archive_existing_versions=True,
    )


def test_register_model_to_staging_swallows_exceptions():
    """Registry failures must be non-fatal — caller must see None, no raise."""
    from fraud_detection.utils.mlflow_utils import register_model_to_staging

    with patch("mlflow.register_model", side_effect=RuntimeError("server down")):
        result = register_model_to_staging(
            artifact_uri="runs:/abc/model_bundle/model_bundle.joblib",
            registered_name="fraud_detection_hybrid",
        )
    assert result is None


def test_pusher_registers_to_staging_on_gate_pass(tmp_path):
    """End-to-end: ModelPusher with gate_passed=True should call the registry
    helper and persist registry coordinates into both serving_manifest.json
    and promotion_metadata.json."""
    from fraud_detection.components.model_pusher import ModelPusher

    training_artifact, eval_artifact = _make_artifacts(tmp_path)
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    config = _pusher_config(current_dir)

    fake_registry_result = {
        "name": "fraud_detection_hybrid",
        "version": "3",
        "stage": "Staging",
        "run_id": "run-abc",
    }

    with patch(
        "fraud_detection.components.model_pusher.ModelPusher._register_to_staging",
        return_value=fake_registry_result,
    ) as mock_register:
        pusher = ModelPusher(config, training_artifact, eval_artifact)
        artifact = pusher.initiate_model_pusher()

    assert artifact.promoted is True
    assert artifact.registered_model_name == "fraud_detection_hybrid"
    assert artifact.registered_model_version == "3"
    assert artifact.registered_model_stage == "Staging"
    mock_register.assert_called_once()

    manifest = json.loads((current_dir / "serving_manifest.json").read_text())
    assert manifest["mlflow_registry"]["version"] == "3"
    assert manifest["mlflow_registry"]["stage"] == "Staging"

    metadata = json.loads((current_dir / "promotion_metadata.json").read_text())
    assert metadata["mlflow_registry"]["version"] == "3"


def test_pusher_does_not_register_when_disabled(tmp_path):
    """register_on_promotion=False should skip the registry path entirely
    and leave the manifest free of any registry block."""
    from fraud_detection.components.model_pusher import ModelPusher

    training_artifact, eval_artifact = _make_artifacts(tmp_path)
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    config = _pusher_config(current_dir, register_on_promotion=False)

    with patch(
        "fraud_detection.components.model_pusher.ModelPusher._register_to_staging",
    ) as mock_register:
        pusher = ModelPusher(config, training_artifact, eval_artifact)
        artifact = pusher.initiate_model_pusher()

    assert artifact.promoted is True
    assert artifact.registered_model_name is None
    mock_register.assert_not_called()

    manifest = json.loads((current_dir / "serving_manifest.json").read_text())
    assert "mlflow_registry" not in manifest


def test_pusher_promotes_even_when_registry_fails(tmp_path):
    """The whole point of making the registry best-effort: a registry hiccup
    must NOT block filesystem promotion. The bundle, manifest, and metadata
    must still be written; only the registry section is absent."""
    from fraud_detection.components.model_pusher import ModelPusher

    training_artifact, eval_artifact = _make_artifacts(tmp_path)
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    config = _pusher_config(current_dir)

    with patch(
        "fraud_detection.components.model_pusher.ModelPusher._register_to_staging",
        return_value=None,
    ):
        pusher = ModelPusher(config, training_artifact, eval_artifact)
        artifact = pusher.initiate_model_pusher()

    assert artifact.promoted is True
    assert artifact.registered_model_name is None

    bundle = current_dir / "model_bundle.joblib"
    manifest_path = current_dir / "serving_manifest.json"
    metadata_path = current_dir / "promotion_metadata.json"
    assert bundle.exists()
    assert manifest_path.exists()
    assert metadata_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert "mlflow_registry" not in manifest


def test_pusher_skips_registry_when_gate_fails(tmp_path):
    """Gate-fail must short-circuit before any registry work happens."""
    from fraud_detection.components.model_pusher import ModelPusher

    training_artifact, eval_artifact = _make_artifacts(tmp_path)
    eval_artifact = ModelEvaluationArtifact(
        scored_players_path=eval_artifact.scored_players_path,
        capture_rate_table_path=eval_artifact.capture_rate_table_path,
        evaluation_report_path=eval_artifact.evaluation_report_path,
        gate_passed=False,
        combined_oos_capture_rate_top_5pct=0.10,
        combined_oos_lift_top_5pct=2.0,
        combined_oos_top_20pct=5,
    )
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    config = _pusher_config(current_dir)

    with patch(
        "fraud_detection.components.model_pusher.ModelPusher._register_to_staging",
    ) as mock_register:
        pusher = ModelPusher(config, training_artifact, eval_artifact)
        artifact = pusher.initiate_model_pusher()

    assert artifact.promoted is False
    mock_register.assert_not_called()
    assert not (current_dir / "model_bundle.joblib").exists()
    assert not (current_dir / "serving_manifest.json").exists()
