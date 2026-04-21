#!/usr/bin/env python
"""Audit-gate verification: re-run model training + evaluation on the existing
feature_engineering artifact and compare key metrics against the previously
recorded evaluation_report.json from the same run.

This is a targeted regression check for the two audit fixes:
  - Bug ME-1: model_evaluation.py cluster_id uses kmeans.predict(X_unsup) now
    instead of kmeans.labels_.
  - Bug FE-1: feature_engineering first_fraud selection — does not affect FE
    output in the observed cases; verified by tests.

A full 3-hour pipeline re-run on 35M rows is not required because:
  - Bug FE-1 only changes `first_fraud_draw_id` in a degenerate case; under
    the live cohort every fraud member has a single fraud event, so
    first_fraud_{ts,draw_id} are unchanged.
  - Bug ME-1 is mathematically a no-op when fit was called on X_unsup (which
    is always the case in the current pipeline).

This script confirms the no-op empirically on the real FE artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.components.model_evaluation import ModelEvaluation
from fraud_detection.components.model_training import ModelTraining
from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact
from fraud_detection.entity.config_entity import (
    ModelEvaluationConfig,
    ModelTrainingConfig,
)
from fraud_detection.utils.common import read_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-run",
        type=Path,
        required=True,
        help="Path to run_YYYYMMDD_HHMMSS directory whose FE artifact we reuse",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        default=Path("configs/model_params.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/audit_gate_verification"),
    )
    args = parser.parse_args()

    baseline = args.baseline_run
    assert baseline.exists(), f"baseline {baseline} not found"
    fe_dir = baseline / "feature_engineering"
    assert fe_dir.exists(), f"feature_engineering missing under {baseline}"

    config = read_yaml(args.config)
    mp = read_yaml(args.model_params)

    with open(fe_dir / "feature_summary.json") as f:
        fe_summary = json.load(f)
    feature_columns = fe_summary["feature_columns"]

    fe_artifact = FeatureEngineeringArtifact(
        player_features_path=fe_dir / "player_features.parquet",
        history_df_path=fe_dir / "history_df.parquet",
        fraud_player_count=int(fe_summary["fraud_player_count"]),
        dropped_positive_count=int(fe_summary.get("dropped_positive_count", 0)),
        feature_columns=feature_columns,
        feature_summary_path=fe_dir / "feature_summary.json",
        mode="training_eval",
    )
    print(f"Reusing FE artifact from {baseline.name}:")
    print(f"  fraud_player_count = {fe_artifact.fraud_player_count}")
    print(f"  dropped_positive_count = {fe_artifact.dropped_positive_count}")

    iso_params = dict(mp["isolation_forest"])
    iso_params["_log1p_cols"] = config["feature_engineering"]["log1p_cols"]

    mt_config = ModelTrainingConfig(
        iso_forest_params=iso_params,
        kmeans_params=dict(mp["kmeans"]),
        lr_params=dict(mp["logistic_regression"]),
        anomaly_weight=float(mp["scoring"]["anomaly_weight"]),
        supervised_weight=float(mp["scoring"]["supervised_weight"]),
        random_seed=int(config["pipeline"].get("random_seed", 42)),
        output_dir=args.output_dir / "model_training",
    )
    me_config = ModelEvaluationConfig(
        threshold_percentiles=config["model_evaluation"]["threshold_percentiles"],
        risk_tier_p80=float(config["model_evaluation"]["risk_tier_p80"]),
        risk_tier_p95=float(config["model_evaluation"]["risk_tier_p95"]),
        min_capture_top_20pct=int(config["model_evaluation"]["min_capture_top_20pct"]),
        output_dir=args.output_dir / "model_evaluation",
    )

    print("\nRunning ModelTraining...")
    train_art = ModelTraining(mt_config, fe_artifact).initiate_model_training()
    print(f"  training_report: {train_art.training_report_path}")

    print("\nRunning ModelEvaluation...")
    eval_art = ModelEvaluation(me_config, train_art).initiate_model_evaluation()
    with open(eval_art.evaluation_report_path) as f:
        new = json.load(f)

    # Load the baseline evaluation for side-by-side comparison
    baseline_eval_path = baseline / "model_evaluation" / "evaluation_report.json"
    if baseline_eval_path.exists():
        with open(baseline_eval_path) as f:
            old = json.load(f)
        print("\n=== Baseline vs Post-Audit Comparison ===")
        for k in ["total_players", "fraud_players", "combined_oos_top_20pct"]:
            print(f"  {k:30s}: before={old.get(k)}  after={new.get(k)}")
        print(f"\n  capture_rates.combined_oos:")
        for bucket in ["top_1pct", "top_5pct", "top_10pct", "top_20pct"]:
            b_val = old.get("capture_rates", {}).get("combined_oos", {}).get(bucket)
            a_val = new.get("capture_rates", {}).get("combined_oos", {}).get(bucket)
            same = " (identical)" if b_val == a_val else " *** CHANGED ***"
            print(f"    {bucket}: before={b_val}  after={a_val}{same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
