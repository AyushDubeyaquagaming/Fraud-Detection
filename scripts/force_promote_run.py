#!/usr/bin/env python
"""Force-promote a completed training run, bypassing the quality gate.

Intended for local dev/testing ONLY. Runs ModelPusher (gate_passed=True),
Monitoring, and BatchScoring against the specified run's artifacts.

Usage:
    python scripts/force_promote_run.py --run-id run_20260427_175142
    python scripts/force_promote_run.py  # uses the most recent run automatically
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from fraud_detection.constants.constants import (
    CONFIG_FILE_PATH,
    BATCH_SCORING_CONFIG_FILE_PATH,
    REPO_ROOT,
)
from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureEngineeringArtifact,
    ModelEvaluationArtifact,
    ModelTrainingArtifact,
)
from fraud_detection.entity.config_entity import ModelPusherConfig, MonitoringConfig
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import read_yaml, read_json

logger = get_logger("force_promote")


def _find_latest_run(artifact_root: Path) -> Path:
    runs_dir = artifact_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory at {runs_dir}")
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No run directories found under artifacts/runs/")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Force-promote a training run (dev only)")
    parser.add_argument("--run-id", default=None, help="Run ID to promote (e.g. run_20260427_175142). Defaults to most recent run.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE_PATH)
    args = parser.parse_args()

    config_dict = read_yaml(args.config)
    artifact_root = REPO_ROOT / config_dict["pipeline"]["artifact_root"]
    current_dir = REPO_ROOT / config_dict["pipeline"]["current_dir"]
    eval_cfg = config_dict["model_evaluation"]
    serving_cfg = config_dict.get("serving", {})
    mlflow_cfg = config_dict.get("mlflow", {})
    mon_cfg_raw = config_dict.get("monitoring", {})

    if args.run_id:
        run_dir = artifact_root / "runs" / args.run_id
        if not run_dir.exists():
            print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = _find_latest_run(artifact_root)
        print(f"No --run-id specified. Using most recent run: {run_dir.name}")

    print(f"\n{'='*60}")
    print(f"Force-promoting run: {run_dir.name}")
    print(f"Artifacts: {run_dir}")
    print(f"Target:    {current_dir}")
    print(f"{'='*60}\n")

    # ── Reconstruct artifacts from disk ──────────────────────────────
    train_dir = run_dir / "model_training"
    eval_dir = run_dir / "model_evaluation"
    fe_dir = run_dir / "feature_engineering"
    ingest_dir = run_dir / "data_ingestion"

    for path, label in [
        (train_dir, "model_training"), (eval_dir, "model_evaluation"),
        (fe_dir, "feature_engineering"), (ingest_dir, "data_ingestion"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} directory missing: {path}", file=sys.stderr)
            return 1

    training_report = read_json(train_dir / "training_report.json")
    eval_report = read_json(eval_dir / "evaluation_report.json")

    training_artifact = ModelTrainingArtifact(
        iso_forest_path=train_dir / "iso_forest.joblib",
        kmeans_path=train_dir / "kmeans.joblib",
        mahalanobis_stats_path=train_dir / "mahalanobis_stats.joblib",
        scaler_path=train_dir / "scaler.joblib",
        lr_operational_path=train_dir / "logistic_regression.joblib",
        training_report_path=train_dir / "training_report.json",
        feature_columns=training_report["feature_columns"],
    )

    # Force gate_passed=True — this is the whole point of this script.
    eval_artifact = ModelEvaluationArtifact(
        scored_players_path=eval_dir / "scored_players.parquet",
        capture_rate_table_path=eval_dir / "capture_rate_table.csv",
        evaluation_report_path=eval_dir / "evaluation_report.json",
        gate_passed=True,  # <-- forced
        combined_oos_capture_rate_top_5pct=float(eval_report.get("combined_oos_capture_rate_top_5pct", 0.0)),
        combined_oos_lift_top_5pct=float(eval_report.get("combined_oos_lift_top_5pct", 0.0)),
        combined_oos_top_20pct=int(eval_report.get("combined_oos_top_20pct", 0)),
    )

    ingest_report = read_json(ingest_dir / "ingestion_report.json")
    ingestion_artifact = DataIngestionArtifact(
        raw_data_path=ingest_dir / "raw_data.parquet",
        ingestion_report_path=ingest_dir / "ingestion_report.json",
        row_count=int(ingest_report.get("row_count", 0)),
        member_count=int(ingest_report.get("member_count", 0)),
        source_type=str(ingest_report.get("source_type", "parquet")),
        strategy_used=ingest_report.get("strategy_used"),
        query_count=int(ingest_report.get("query_count", 1)),
        date_range=ingest_report.get("date_range"),
    )

    fe_summary = read_json(fe_dir / "feature_summary.json") if (fe_dir / "feature_summary.json").exists() else {}
    fe_artifact = FeatureEngineeringArtifact(
        player_features_path=fe_dir / "player_features.parquet",
        history_df_path=fe_dir / "history_df.parquet",
        fraud_player_count=int(fe_summary.get("fraud_player_count", eval_report.get("fraud_players", 0))),
        dropped_positive_count=int(fe_summary.get("dropped_positive_count", 0)),
        feature_columns=training_report["feature_columns"],
        feature_summary_path=fe_dir / "feature_summary.json",
        mode="training_eval",
    )

    # ── MLflow setup (non-fatal) ──────────────────────────────────────
    import mlflow
    from fraud_detection.utils.mlflow_utils import get_tracking_uri, setup_mlflow

    tracking_uri = get_tracking_uri()
    experiment_name = mlflow_cfg.get("experiment_name", "fraud_detection_hybrid")
    exp_id = setup_mlflow(tracking_uri, experiment_name)

    mlflow_active = False
    try:
        mlflow.start_run(run_name=f"force_promote_{run_dir.name}", experiment_id=exp_id)
        mlflow_active = True
        mlflow.set_tag("force_promoted", "true")
        mlflow.set_tag("source_run_id", run_dir.name)
        mlflow.set_tag("source", "force_promote_run")
        # Log the key metrics from the original evaluation so this run is searchable
        mlflow.log_metrics({
            "combined_oos_capture_rate_top_5pct": eval_artifact.combined_oos_capture_rate_top_5pct,
            "combined_oos_lift_top_5pct": eval_artifact.combined_oos_lift_top_5pct,
            "gate_passed": 1,
        })
    except Exception as exc:
        logger.warning("MLflow run could not start: %s — continuing without MLflow", exc)

    # ── Step 1: Monitoring (BEFORE Pusher — matches training_pipeline.py) ──
    # This must run before ModelPusher overwrites promotion_metadata.json.
    # Otherwise the new metadata points at THIS run, monitoring tries to compare
    # the run to itself, and either skips (with the self-reference guard) or
    # — without the guard — loads the same 44M-row parquet twice and OOMs.
    print("[1/3] Running Monitoring (Evidently drift reports)...")
    from fraud_detection.components.monitoring import Monitoring

    monitoring_config = MonitoringConfig(
        enabled=bool(mon_cfg_raw.get("enabled", True)),
        reports_dir=str(mon_cfg_raw.get("reports_dir", "monitoring")),
        sample_size=int(mon_cfg_raw.get("sample_size", 50000)),
        monitored_features=list(mon_cfg_raw.get("monitored_features", [])),
        drift_threshold=float(mon_cfg_raw.get("drift_threshold", 0.3)),
        reference_from_current_metadata=bool(mon_cfg_raw.get("reference_from_current_metadata", True)),
    )

    mon_artifact = Monitoring(
        config=monitoring_config,
        current_dir=current_dir,
        ingestion_artifact=ingestion_artifact,
        fe_artifact=fe_artifact,
        eval_artifact=eval_artifact,
        run_dir=run_dir,
    ).initiate_monitoring()

    if mon_artifact.monitoring_completed:
        print(f"  Drift reports written to: {mon_artifact.reports_dir}")
        if mon_artifact.drift_summary_path:
            with open(mon_artifact.drift_summary_path) as f:
                summary = json.load(f)
            print(f"  Overall drift detected: {summary.get('overall_drift_detected', 'unknown')}")
        if mlflow_active and mon_artifact.reports_dir:
            from fraud_detection.utils.mlflow_utils import log_artifacts_safe
            log_artifacts_safe(str(mon_artifact.reports_dir))
    else:
        print("  Monitoring skipped — no valid reference run in promotion_metadata.json.")
        print("  Expected on first successful promotion. Next run will produce drift reports.")

    # ── Step 2: ModelPusher (forced gate pass) ───────────────────────
    print("\n[2/3] Running ModelPusher (gate forced to passed)...")
    from fraud_detection.components.model_pusher import ModelPusher

    pusher_config = ModelPusherConfig(
        current_dir=current_dir,
        manifest_file=str(serving_cfg.get("manifest_file", "serving_manifest.json")),
        model_version=str(serving_cfg.get("model_version", "hybrid_v1")),
        min_capture_rate_top_5pct=float(eval_cfg.get("min_capture_rate_top_5pct", 0.10)),
        min_lift_top_5pct=float(eval_cfg.get("min_lift_top_5pct", 2.0)),
        register_on_promotion=bool(mlflow_cfg.get("register_on_promotion", True)),
        registered_model_name=str(mlflow_cfg.get("registered_model_name", "fraud_detection_hybrid")),
        archive_existing_staging=bool(mlflow_cfg.get("archive_existing_staging", True)),
        auto_promote_to_production=False,
    )

    pusher_artifact = ModelPusher(pusher_config, training_artifact, eval_artifact).initiate_model_pusher()
    if not pusher_artifact.promoted:
        print("ERROR: ModelPusher still did not promote — check logs above.", file=sys.stderr)
        if mlflow_active:
            mlflow.end_run(status="FAILED")
        return 1

    print(f"  Promoted: {pusher_artifact.promoted}")
    if pusher_artifact.registered_model_version:
        print(f"  MLflow Registry: {pusher_artifact.registered_model_name} v{pusher_artifact.registered_model_version} → {pusher_artifact.registered_model_stage}")

    # ── Step 3: Weekly serving snapshot (batch scoring) ──────────────
    print("\n[3/3] Running BatchScoring (weekly serving snapshot)...")
    from fraud_detection.pipeline.batch_scoring_pipeline import BatchScoringPipeline

    output_dir = BatchScoringPipeline(config_path=BATCH_SCORING_CONFIG_FILE_PATH).run()
    print(f"  Weekly snapshot written to: {output_dir}")
    print(f"  artifacts/current/ now has hybrid_scored_players.parquet + alert_queue.csv")

    if mlflow_active:
        from fraud_detection.utils.mlflow_utils import log_artifact_safe
        log_artifact_safe(str(pusher_artifact.model_bundle_path))
        mlflow.set_tag("promoted", "true")
        mlflow.end_run(status="FINISHED")

    print(f"\n{'='*60}")
    print(f"Force-promotion complete.")
    print(f"  Source run:    {run_dir.name}")
    print(f"  Bundle:        {current_dir / 'model_bundle.joblib'}")
    print(f"  Manifest:      {current_dir / 'serving_manifest.json'}")
    print(f"  Scored parquet:{current_dir / 'hybrid_scored_players.parquet'}")
    print(f"\nNext steps:")
    print(f"  - View MLflow:    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000")
    print(f"  - Start API:      python scripts/run_api.py  (then /admin/reload if already running)")
    print(f"  - Streamlit:      python -m streamlit run streamlit_hybrid_demo.py")
    if mon_artifact.monitoring_completed and mon_artifact.reports_dir:
        print(f"  - Evidently:      open {mon_artifact.reports_dir / 'data_drift.html'} in browser")
    else:
        print(f"  - Evidently:      will work on NEXT training run (reference is now set)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
