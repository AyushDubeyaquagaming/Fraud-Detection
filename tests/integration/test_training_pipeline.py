from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

# Ensure src is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from fraud_detection.constants.constants import CONFIG_FILE_PATH, REPO_ROOT


PARQUET_PATH = REPO_ROOT / "data_cache" / "fraud_modeling_pull.parquet"

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="data_cache/fraud_modeling_pull.parquet not found — skipping integration test",
)


def test_full_training_pipeline(tmp_path):
    """Run end-to-end training pipeline on cached parquet and verify all outputs."""
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data_ingestion"]["source"] = "parquet"
    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    run_dir = TrainingPipeline(config_path=temp_config_path).run()

    assert run_dir.exists(), f"run_dir does not exist: {run_dir}"

    # --- Data ingestion outputs ---
    assert (run_dir / "data_ingestion" / "raw_data.parquet").exists()
    assert (run_dir / "data_ingestion" / "ingestion_report.json").exists()
    with open(run_dir / "data_ingestion" / "ingestion_report.json") as f:
        ing_report = json.load(f)
    assert ing_report["row_count"] > 0
    assert ing_report["member_count"] > 0

    # --- Data validation outputs ---
    assert (run_dir / "data_validation" / "validation_report.json").exists()
    with open(run_dir / "data_validation" / "validation_report.json") as f:
        val_report = json.load(f)
    assert val_report["all_passed"] is True

    # --- Feature engineering outputs ---
    assert (run_dir / "feature_engineering" / "player_features.parquet").exists()
    assert (run_dir / "feature_engineering" / "history_df.parquet").exists()
    assert (run_dir / "feature_engineering" / "feature_summary.json").exists()
    with open(run_dir / "feature_engineering" / "feature_summary.json") as f:
        fe_summary = json.load(f)
    assert fe_summary["player_count"] > 0
    assert fe_summary["fraud_player_count"] > 0

    # --- Model training outputs ---
    for fname in ["iso_forest.joblib", "kmeans.joblib", "mahalanobis_stats.joblib",
                  "scaler.joblib", "logistic_regression.joblib", "training_report.json"]:
        assert (run_dir / "model_training" / fname).exists(), f"Missing: {fname}"
    with open(run_dir / "model_training" / "training_report.json") as f:
        tr = json.load(f)
    assert tr["pr_auc"] > 0
    assert tr["roc_auc"] > 0.5

    # --- Model evaluation outputs ---
    assert (run_dir / "model_evaluation" / "scored_players.parquet").exists()
    assert (run_dir / "model_evaluation" / "capture_rate_table.csv").exists()
    assert (run_dir / "model_evaluation" / "evaluation_report.json").exists()
    with open(run_dir / "model_evaluation" / "evaluation_report.json") as f:
        ev = json.load(f)
    assert "capture_rates" in ev
    assert "gate_passed" in ev

    # --- Run metadata ---
    assert (run_dir / "run_metadata.json").exists()
    with open(run_dir / "run_metadata.json") as f:
        meta = json.load(f)
    assert meta["status"] == "FINISHED"

    # --- Promotion gate & current/ ---
    if ev["gate_passed"]:
        current_dir = REPO_ROOT / "artifacts" / "current"
        assert (current_dir / "model_bundle.joblib").exists(), "model_bundle.joblib missing"
        bundle = joblib.load(current_dir / "model_bundle.joblib")
        assert "iso_forest" in bundle
        assert "lr_operational" in bundle
        assert "feature_columns" in bundle
        assert "log1p_columns" in bundle
        assert "style_pca" in bundle
        assert (current_dir / "hybrid_scored_players.parquet").exists()
        assert (current_dir / "alert_queue.csv").exists()
        assert (current_dir / "promotion_metadata.json").exists()

    # --- Schema checks on scored output ---
    scored = pd.read_parquet(run_dir / "model_evaluation" / "scored_players.parquet")
    for col in [
        "member_id", "risk_score", "anomaly_score", "supervised_score", "risk_tier",
        "cluster_id", "style_pc1", "style_pc2",
    ]:
        assert col in scored.columns, f"Missing column in scored_players: {col}"
    assert scored["risk_score"].between(0, 10).all(), "risk_score out of expected range"
    assert (run_dir / "model_evaluation" / "plots" / "feature_importance.png").exists()
    assert (run_dir / "model_evaluation" / "plots" / "confusion_matrix.png").exists()
    assert (run_dir / "model_evaluation" / "plots" / "correlation_heatmap.png").exists()
