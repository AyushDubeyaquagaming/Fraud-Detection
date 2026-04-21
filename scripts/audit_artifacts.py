#!/usr/bin/env python
"""Audit script: verify a promoted artifact directory is internally consistent.

Usage:
    python scripts/audit_artifacts.py
    python scripts/audit_artifacts.py --config configs/config.yaml
    python scripts/audit_artifacts.py --artifact-dir artifacts/current
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from fraud_detection.constants.constants import CONFIG_FILE_PATH, CURRENT_DIR, REPO_ROOT
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import read_yaml

logger = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(name: str, condition: bool, detail: str = "") -> dict:
    status = PASS if condition else FAIL
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    if condition:
        logger.info(msg)
    else:
        logger.error(msg)
    return {"check": name, "status": status, "detail": detail}


def warn(name: str, detail: str = "") -> dict:
    msg = f"[{WARN}] {name}"
    if detail:
        msg += f" — {detail}"
    logger.warning(msg)
    return {"check": name, "status": WARN, "detail": detail}


def _resolve_artifact_dir(config_path: Path, artifact_dir_arg: str | None) -> Path:
    if artifact_dir_arg:
        artifact_dir = Path(artifact_dir_arg)
        if not artifact_dir.is_absolute():
            artifact_dir = REPO_ROOT / artifact_dir
        return artifact_dir

    config = read_yaml(config_path)
    configured_dir = config.get("pipeline", {}).get("current_dir")
    if configured_dir:
        configured_path = Path(configured_dir)
        return configured_path if configured_path.is_absolute() else REPO_ROOT / configured_path
    return CURRENT_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a promoted artifact directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE_PATH,
        help="Path to config.yaml used to resolve pipeline.current_dir.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Explicit artifact directory to audit. Overrides pipeline.current_dir.",
    )
    args = parser.parse_args()
    artifact_dir = _resolve_artifact_dir(args.config, args.artifact_dir)

    print("\n" + "=" * 70)
    print("Fraud Detection Artifact Audit")
    print("=" * 70)

    results = []

    # 1. Current directory exists
    results.append(check("current_dir_exists", artifact_dir.exists(), str(artifact_dir)))

    # 2. model_bundle.joblib exists and loads
    bundle_path = artifact_dir / "model_bundle.joblib"
    bundle_exists = bundle_path.exists()
    results.append(check("model_bundle_exists", bundle_exists, str(bundle_path)))

    if bundle_exists:
        try:
            bundle = joblib.load(bundle_path)
            required_keys = [
                "iso_forest", "kmeans", "mahal_stats", "scaler_unsup", "scaler_operational",
                "lr_operational", "feature_columns", "iso_min", "iso_max",
                "mahal_min", "mahal_max", "cluster_min", "cluster_max",
                "risk_p80", "risk_p95", "anomaly_weight", "supervised_weight",
                "log1p_columns", "anomaly_component_weights",
            ]
            missing = [k for k in required_keys if k not in bundle]
            results.append(check("model_bundle_keys", len(missing) == 0, f"missing={missing}"))
            results.append(check("feature_columns_nonempty", len(bundle.get("feature_columns", [])) > 0))
            results.append(check("style_models_available", bundle.get("style_scaler") is not None and bundle.get("style_pca") is not None))
        except Exception as e:
            results.append(check("model_bundle_loads", False, str(e)))

    # 3. hybrid_scored_players.parquet exists and has required columns
    scored_path = artifact_dir / "hybrid_scored_players.parquet"
    scored_exists = scored_path.exists()
    results.append(check("scored_players_exists", scored_exists))
    if scored_exists:
        try:
            scored = pd.read_parquet(scored_path)
            required_cols = [
                "member_id", "risk_score", "anomaly_score", "supervised_score", "risk_tier",
                "cluster_id", "style_pc1", "style_pc2",
            ]
            missing_cols = [c for c in required_cols if c not in scored.columns]
            results.append(check("scored_players_columns", len(missing_cols) == 0, f"missing={missing_cols}"))
            results.append(check("scored_players_rows", len(scored) > 0, f"rows={len(scored)}"))
        except Exception as e:
            results.append(check("scored_players_reads", False, str(e)))

    # 4. alert_queue.csv exists
    alert_path = artifact_dir / "alert_queue.csv"
    alert_exists = alert_path.exists()
    results.append(check("alert_queue_exists", alert_exists))
    if alert_exists:
        try:
            alert_df = pd.read_csv(alert_path)
            results.append(check("alert_queue_nonempty", len(alert_df) > 0, f"rows={len(alert_df)}"))
        except Exception as e:
            results.append(check("alert_queue_reads", False, str(e)))

    # 5. hybrid_evaluation.json exists and has expected fields
    eval_path = artifact_dir / "hybrid_evaluation.json"
    eval_exists = eval_path.exists()
    results.append(check("evaluation_json_exists", eval_exists))
    if eval_exists:
        try:
            with open(eval_path) as f:
                ev = json.load(f)
            results.append(check("evaluation_has_capture_rates", "capture_rates" in ev or "score_distribution" in ev))
            results.append(check("evaluation_cohort_note", "cohort_scope_note" in ev))
        except Exception as e:
            results.append(check("evaluation_json_reads", False, str(e)))

    # 6. promotion_metadata.json
    promo_path = artifact_dir / "promotion_metadata.json"
    promo_exists = promo_path.exists()
    results.append(check("promotion_metadata_exists", promo_exists))
    if promo_exists:
        try:
            with open(promo_path) as f:
                promo = json.load(f)
            results.append(check("promotion_gate_passed", promo.get("gate_passed", False), str(promo.get("gate_passed"))))
            # Phase 2 rebaseline — report new primary gate metrics when present.
            cap5 = promo.get("combined_oos_capture_rate_top_5pct")
            lift5 = promo.get("combined_oos_lift_top_5pct")
            if cap5 is not None or lift5 is not None:
                results.append(
                    warn(
                        "promotion_gate_metrics",
                        f"capture_rate_top_5pct={cap5} lift_top_5pct={lift5}",
                    )
                )
        except Exception as e:
            results.append(check("promotion_metadata_reads", False, str(e)))

    # 7. hybrid_inference.py reads from artifacts/current/ or data_cache/
    inference_path = REPO_ROOT / "hybrid_inference.py"
    if inference_path.exists():
        content = inference_path.read_text()
        uses_current = "artifacts/current" in content or "CURRENT_DIR" in content
        results.append(
            check("hybrid_inference_updated", uses_current)
            if uses_current
            else warn("hybrid_inference_uses_legacy_path", "Still reading from data_cache/ — migration pending")
        )
    else:
        results.append(warn("hybrid_inference_not_found"))

    # Summary
    print("\n" + "-" * 70)
    failed = [r for r in results if r["status"] == FAIL]
    warned = [r for r in results if r["status"] == WARN]
    passed = [r for r in results if r["status"] == PASS]
    print(f"Results: {len(passed)} passed, {len(warned)} warnings, {len(failed)} failed")
    if failed:
        print("\nFailed checks:")
        for r in failed:
            print(f"  - {r['check']}: {r['detail']}")
    if warned:
        print("\nWarnings:")
        for r in warned:
            print(f"  - {r['check']}: {r['detail']}")
    print("=" * 70 + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
