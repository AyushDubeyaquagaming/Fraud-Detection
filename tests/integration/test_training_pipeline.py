from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Ensure src is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from fraud_detection.constants.constants import CONFIG_FILE_PATH, REPO_ROOT


def _make_bets(numbers: list[int], amounts: list[float]) -> str:
    return json.dumps(
        [
            {"number": str(number), "bet_amount": amount}
            for number, amount in zip(numbers, amounts)
        ]
    )


def _build_synthetic_raw_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    fraud_rows: list[dict] = []
    current_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    draw_id = 1

    for member_idx in range(40):
        member_id = f"N{member_idx:03d}"
        ccs_id = f"CCS_{member_idx % 10:02d}"
        for draw_idx in range(25):
            bet_amounts = [4.0 + (draw_idx % 3), 3.0 + (draw_idx % 2), 2.0]
            total_bet_amount = sum(bet_amounts)
            rows.append(
                {
                    "member_id": member_id,
                    "draw_id": draw_id,
                    "bets": _make_bets(
                        [
                            (draw_idx * 3) % 38,
                            (draw_idx * 7 + 5) % 38,
                            (draw_idx * 11 + 9) % 38,
                        ],
                        bet_amounts,
                    ),
                    "win_points": total_bet_amount * (0.75 + 0.05 * (draw_idx % 4)),
                    "total_bet_amount": total_bet_amount,
                    "session_id": draw_idx // 5,
                    "ccs_id": ccs_id,
                    "createdAt": current_ts,
                    "updatedAt": current_ts + timedelta(seconds=30),
                    "trans_date": current_ts,
                }
            )
            draw_id += 1
            current_ts += timedelta(minutes=1)

    for member_idx in range(8):
        member_id = f"F{member_idx:03d}"
        ccs_id = f"CCS_F{member_idx % 2}"
        for draw_idx in range(30):
            bet_amounts = [50.0, 45.0 + float(draw_idx % 2), 40.0]
            total_bet_amount = sum(bet_amounts)
            current_draw_id = draw_id
            rows.append(
                {
                    "member_id": member_id,
                    "draw_id": current_draw_id,
                    "bets": _make_bets([17, 18, 19], bet_amounts),
                    "win_points": 5.0 if draw_idx % 3 else 0.0,
                    "total_bet_amount": total_bet_amount,
                    "session_id": draw_idx // 10,
                    "ccs_id": ccs_id,
                    "createdAt": current_ts,
                    "updatedAt": current_ts + timedelta(seconds=15),
                    "trans_date": current_ts,
                }
            )
            if draw_idx == 20:
                fraud_rows.append({"member_id": member_id, "draw_id": current_draw_id})
            draw_id += 1
            current_ts += timedelta(minutes=1)

    return pd.DataFrame(rows), pd.DataFrame(fraud_rows)


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_df, fraud_df = _build_synthetic_raw_df()
    parquet_path = tmp_path / "synthetic_raw.parquet"
    fraud_csv_path = tmp_path / "synthetic_fraud.csv"
    raw_df.to_parquet(parquet_path, index=False)
    fraud_df.to_csv(fraud_csv_path, index=False)
    return parquet_path, fraud_csv_path


def test_full_training_pipeline(tmp_path):
    """Run end-to-end training pipeline on cached parquet and verify all outputs."""
    parquet_path, fraud_csv_path = _write_synthetic_inputs(tmp_path)

    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data_ingestion"]["source"] = "parquet"
    config["data_ingestion"]["parquet_path"] = str(parquet_path)
    config["data_validation"]["fraud_csv_path"] = str(fraud_csv_path)
    config["pipeline"]["artifact_root"] = str(tmp_path / "artifacts")
    config["pipeline"]["current_dir"] = str(tmp_path / "artifacts" / "current")
    # Use synthetic-data-appropriate gate thresholds so the test exercises the
    # promoted-artifact code path without requiring an unrealistically strong
    # model on the 48-player toy cohort.
    config["data_validation"]["min_row_count"] = 100
    config["model_evaluation"]["min_capture_rate_top_5pct"] = 0.0
    config["model_evaluation"]["min_lift_top_5pct"] = 0.0
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
    assert "capture_stats" in ev
    assert "gate_passed" in ev
    assert "combined_oos_capture_rate_top_5pct" in ev
    assert "combined_oos_lift_top_5pct" in ev
    # capture_stats has per-bucket lift / precision / capture_rate
    combined_oos_stats = ev["capture_stats"]["combined_oos"]
    for bucket in ["top_5pct", "top_50", "top_500"]:
        assert bucket in combined_oos_stats, f"missing {bucket} in combined_oos capture_stats"
        assert "lift" in combined_oos_stats[bucket]
        assert "capture_rate" in combined_oos_stats[bucket]

    # --- Run metadata ---
    assert (run_dir / "run_metadata.json").exists()
    with open(run_dir / "run_metadata.json") as f:
        meta = json.load(f)
    assert meta["status"] == "FINISHED"

    # --- Promotion gate & current/ ---
    if ev["gate_passed"]:
        current_dir = tmp_path / "artifacts" / "current"
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
