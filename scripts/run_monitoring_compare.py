#!/usr/bin/env python
"""Run Monitoring (Evidently drift reports) standalone, comparing two runs.

Compares the CURRENT run against an explicit REFERENCE run, without touching
artifacts/current/ or promotion_metadata.json. Useful for ad-hoc drift
investigations and for visualising Evidently output before a real promotion.

Usage:
    python scripts/run_monitoring_compare.py \\
        --current-run-id  run_20260427_175142 \\
        --reference-run-id run_20260422_105102

The script writes drift reports to:
    artifacts/runs/<current-run>/monitoring/
        - data_drift.html
        - feature_drift.html
        - prediction_drift.html
        - drift_summary.json

It also opens nothing — the user opens the HTMLs in their browser manually.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from fraud_detection.constants.constants import CONFIG_FILE_PATH
from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureEngineeringArtifact,
    ModelEvaluationArtifact,
)
from fraud_detection.entity.config_entity import MonitoringConfig
from fraud_detection.utils.common import read_yaml, read_json


_REQUIRED_ARTIFACTS = {
    "raw":     "data_ingestion/raw_data.parquet",
    "feat":    "feature_engineering/player_features.parquet",
    "scored":  "model_evaluation/scored_players.parquet",
    "ingest":  "data_ingestion/ingestion_report.json",
    "report":  "model_evaluation/evaluation_report.json",
    "training":"model_training/training_report.json",
}


def _validate_run_dir(run_dir: Path, label: str) -> None:
    if not run_dir.exists():
        print(f"ERROR: {label} run does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)
    missing = [p for p in _REQUIRED_ARTIFACTS.values() if not (run_dir / p).exists()]
    if missing:
        print(f"ERROR: {label} run {run_dir.name} is missing required artifacts:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)


def _find_latest_run(runs_dir: Path) -> Path:
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        print(f"ERROR: no run_ directories under {runs_dir}", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two runs via Evidently drift reports")
    parser.add_argument("--current-run-id", default=None,
                        help="Run ID to monitor (default: most recent)")
    parser.add_argument("--reference-run-id", required=True,
                        help="Run ID to use as the reference (the older run to compare against)")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE_PATH)
    args = parser.parse_args()

    config_dict = read_yaml(args.config)
    artifact_root = REPO_ROOT / config_dict["pipeline"]["artifact_root"]
    runs_dir = artifact_root / "runs"
    mon_cfg_raw = config_dict.get("monitoring", {})

    cur_run_dir = (runs_dir / args.current_run_id) if args.current_run_id else _find_latest_run(runs_dir)
    ref_run_dir = runs_dir / args.reference_run_id

    print("=" * 60)
    print("Standalone Evidently Drift Comparison")
    print("=" * 60)
    print(f"Current run:   {cur_run_dir.name}")
    print(f"Reference run: {ref_run_dir.name}")
    print()

    _validate_run_dir(cur_run_dir, "current")
    _validate_run_dir(ref_run_dir, "reference")
    if cur_run_dir.resolve() == ref_run_dir.resolve():
        print("ERROR: current and reference run must be different.", file=sys.stderr)
        return 1

    # ── Reconstruct artifacts for the CURRENT run only.
    # Monitoring uses these directly. The reference run is discovered via the
    # promotion_metadata.json we write into the temp current_dir below.
    train_report = read_json(cur_run_dir / "model_training" / "training_report.json")
    eval_report = read_json(cur_run_dir / "model_evaluation" / "evaluation_report.json")
    ingest_report = read_json(cur_run_dir / "data_ingestion" / "ingestion_report.json")

    ingestion_artifact = DataIngestionArtifact(
        raw_data_path=cur_run_dir / "data_ingestion" / "raw_data.parquet",
        ingestion_report_path=cur_run_dir / "data_ingestion" / "ingestion_report.json",
        row_count=int(ingest_report.get("row_count", 0)),
        member_count=int(ingest_report.get("member_count", 0)),
        source_type=str(ingest_report.get("source_type", "parquet")),
        strategy_used=ingest_report.get("strategy_used"),
        query_count=int(ingest_report.get("query_count", 1)),
        date_range=ingest_report.get("date_range"),
    )

    fe_summary_path = cur_run_dir / "feature_engineering" / "feature_summary.json"
    fe_summary = read_json(fe_summary_path) if fe_summary_path.exists() else {}
    fe_artifact = FeatureEngineeringArtifact(
        player_features_path=cur_run_dir / "feature_engineering" / "player_features.parquet",
        history_df_path=cur_run_dir / "feature_engineering" / "history_df.parquet",
        fraud_player_count=int(fe_summary.get("fraud_player_count", eval_report.get("fraud_players", 0))),
        dropped_positive_count=int(fe_summary.get("dropped_positive_count", 0)),
        feature_columns=train_report["feature_columns"],
        feature_summary_path=fe_summary_path,
        mode="training_eval",
    )

    eval_artifact = ModelEvaluationArtifact(
        scored_players_path=cur_run_dir / "model_evaluation" / "scored_players.parquet",
        capture_rate_table_path=cur_run_dir / "model_evaluation" / "capture_rate_table.csv",
        evaluation_report_path=cur_run_dir / "model_evaluation" / "evaluation_report.json",
        gate_passed=bool(eval_report.get("gate_passed", False)),
        combined_oos_capture_rate_top_5pct=float(eval_report.get("combined_oos_capture_rate_top_5pct", 0.0)),
        combined_oos_lift_top_5pct=float(eval_report.get("combined_oos_lift_top_5pct", 0.0)),
        combined_oos_top_20pct=int(eval_report.get("combined_oos_top_20pct", 0)),
    )

    monitoring_config = MonitoringConfig(
        enabled=True,
        reports_dir=str(mon_cfg_raw.get("reports_dir", "monitoring")),
        sample_size=int(mon_cfg_raw.get("sample_size", 50000)),
        monitored_features=list(mon_cfg_raw.get("monitored_features", [])),
        drift_threshold=float(mon_cfg_raw.get("drift_threshold", 0.3)),
        reference_from_current_metadata=True,
    )

    # ── Use a temp dir as Monitoring's "current_dir" with a synthetic
    # promotion_metadata.json pointing at the chosen reference. This way the
    # real artifacts/current/ stays untouched.
    from fraud_detection.components.monitoring import Monitoring

    with tempfile.TemporaryDirectory(prefix="mon_compare_") as tmp:
        tmp_current = Path(tmp)
        synthetic_meta = {
            "gate_passed": True,
            "run_dir": str(ref_run_dir),
            "source": "run_monitoring_compare",
        }
        (tmp_current / "promotion_metadata.json").write_text(json.dumps(synthetic_meta))

        print("[1/1] Running Monitoring...")
        mon_artifact = Monitoring(
            config=monitoring_config,
            current_dir=tmp_current,
            ingestion_artifact=ingestion_artifact,
            fe_artifact=fe_artifact,
            eval_artifact=eval_artifact,
            run_dir=cur_run_dir,
        ).initiate_monitoring()

    print()
    if mon_artifact.monitoring_completed:
        print("Monitoring completed.")
        print(f"  Reports dir:    {mon_artifact.reports_dir}")
        if mon_artifact.data_drift_report_path:
            print(f"  Data drift:     {mon_artifact.data_drift_report_path}")
        if mon_artifact.feature_drift_report_path:
            print(f"  Feature drift:  {mon_artifact.feature_drift_report_path}")
        if mon_artifact.prediction_drift_report_path:
            print(f"  Prediction drift: {mon_artifact.prediction_drift_report_path}")
        if mon_artifact.drift_summary_path:
            with open(mon_artifact.drift_summary_path) as f:
                summary = json.load(f)
            print(f"  Drift summary:  {mon_artifact.drift_summary_path}")
            print()
            print("=== Drift Summary ===")
            print(f"  Reference run:        {Path(summary['reference_run_dir']).name}")
            print(f"  Overall drift detected: {summary['overall_drift_detected']}")
            print(f"  Average drift share:    {summary['average_drift_share']:.3f}")
            print(f"  Threshold:              {summary['drift_threshold']:.3f}")
            print(f"  Above threshold:        {summary['above_threshold']}")
            print()
            for r in summary.get("reports", []):
                print(f"  - {r['label']}: drift={r['dataset_drift']} "
                      f"share={r['share_of_drifted_columns']:.3f} "
                      f"({r['number_of_drifted_columns']}/{r['number_of_columns']} cols)")
        print()
        print("Open the HTML files in your browser to see the drift visualisations.")
    else:
        print("Monitoring did NOT complete. Check the warnings above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
