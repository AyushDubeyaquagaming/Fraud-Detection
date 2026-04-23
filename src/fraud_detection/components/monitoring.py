from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureEngineeringArtifact,
    ModelEvaluationArtifact,
    MonitoringArtifact,
)
from fraud_detection.entity.config_entity import MonitoringConfig
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, write_json

logger = get_logger(__name__)

_SCORE_COL = "hybrid_score"
_EMPTY = MonitoringArtifact(
    reports_dir=None,
    data_drift_report_path=None,
    feature_drift_report_path=None,
    prediction_drift_report_path=None,
    drift_summary_path=None,
    monitoring_completed=False,
)


def _load_reference_run_dir(current_dir: Path) -> Path | None:
    meta_path = current_dir / "promotion_metadata.json"
    if not meta_path.exists():
        logger.warning("Monitoring: no promotion_metadata.json at %s — skipping", meta_path)
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        run_dir_str = meta.get("run_dir")
        if not run_dir_str:
            logger.warning("Monitoring: promotion_metadata.json missing 'run_dir' — skipping")
            return None
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            logger.warning("Monitoring: reference run_dir does not exist: %s — skipping", run_dir)
            return None
        return run_dir
    except Exception as exc:
        logger.warning("Monitoring: failed to read promotion_metadata.json: %s — skipping", exc)
        return None


def _sample(df: pd.DataFrame, n: int, label_col: str | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df
    if label_col and label_col in df.columns:
        positives = df[df[label_col] == 1]
        if len(positives) >= n:
            return positives.sample(n, random_state=42)

        negative_count = min(n - len(positives), len(df) - len(positives))
        negatives = df[df[label_col] == 0].sample(negative_count, random_state=42)
        return pd.concat([positives, negatives], ignore_index=True)
    return df.sample(n, random_state=42)


def _run_report(ref: pd.DataFrame, cur: pd.DataFrame, out_path: Path, label: str) -> dict[str, Any]:
    from evidently.report import Report
    from evidently.metric_preset.data_drift import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(out_path))

    raw = report.as_dict()
    metrics = raw.get("metrics", [])
    dataset_result = next(
        (m["result"] for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
    )
    return {
        "label": label,
        "report_path": str(out_path),
        "dataset_drift": dataset_result.get("dataset_drift", False),
        "share_of_drifted_columns": dataset_result.get("share_of_drifted_columns", 0.0),
        "number_of_columns": dataset_result.get("number_of_columns", 0),
        "number_of_drifted_columns": dataset_result.get("number_of_drifted_columns", 0),
    }


class Monitoring:
    def __init__(
        self,
        config: MonitoringConfig,
        current_dir: Path,
        ingestion_artifact: DataIngestionArtifact,
        fe_artifact: FeatureEngineeringArtifact,
        eval_artifact: ModelEvaluationArtifact,
        run_dir: Path,
    ):
        self.config = config
        self.current_dir = current_dir
        self.ingestion_artifact = ingestion_artifact
        self.fe_artifact = fe_artifact
        self.eval_artifact = eval_artifact
        self.run_dir = run_dir

    def initiate_monitoring(self) -> MonitoringArtifact:
        if not self.config.enabled:
            logger.info("Monitoring: disabled in config — skipping")
            return _EMPTY

        logger.info("Monitoring: starting")
        try:
            return self._run()
        except Exception as exc:
            logger.warning("Monitoring: non-fatal failure — %s", exc, exc_info=True)
            return _EMPTY

    def _run(self) -> MonitoringArtifact:
        ref_run_dir = _load_reference_run_dir(self.current_dir)
        if ref_run_dir is None:
            return _EMPTY

        reports_dir = self.run_dir / self.config.reports_dir
        ensure_dir(reports_dir)

        n = self.config.sample_size
        monitored = self.config.monitored_features
        threshold = self.config.drift_threshold
        summaries: list[dict[str, Any]] = []

        # --- Data drift ---
        cur_raw_path = self.ingestion_artifact.raw_data_path
        ref_raw_path = ref_run_dir / "data_ingestion" / "raw_data.parquet"
        data_report_path = reports_dir / "data_drift.html"
        feature_report_path = reports_dir / "feature_drift.html"
        prediction_report_path = reports_dir / "prediction_drift.html"

        if cur_raw_path.exists() and ref_raw_path.exists():
            cur_raw = _sample(pd.read_parquet(cur_raw_path), n)
            ref_raw = _sample(pd.read_parquet(ref_raw_path), n)
            shared_cols = [c for c in cur_raw.columns if c in ref_raw.columns and cur_raw[c].dtype.kind in "iuf"]
            if shared_cols:
                summary = _run_report(ref_raw[shared_cols], cur_raw[shared_cols], data_report_path, "data_drift")
                summaries.append(summary)
                logger.info("Monitoring: data_drift — drift=%s share=%.2f", summary["dataset_drift"], summary["share_of_drifted_columns"])
            else:
                logger.warning("Monitoring: no shared numeric columns for data drift — skipping data report")
        else:
            logger.warning("Monitoring: raw data paths missing (cur=%s ref=%s) — skipping data drift", cur_raw_path.exists(), ref_raw_path.exists())

        # --- Feature drift ---
        cur_feat_path = self.fe_artifact.player_features_path
        ref_feat_path = ref_run_dir / "feature_engineering" / "player_features.parquet"

        if cur_feat_path.exists() and ref_feat_path.exists():
            cur_feat = _sample(pd.read_parquet(cur_feat_path), n, "event_fraud_flag")
            ref_feat = _sample(pd.read_parquet(ref_feat_path), n, "event_fraud_flag")
            feat_cols = [c for c in monitored if c in cur_feat.columns and c in ref_feat.columns]
            if feat_cols:
                summary = _run_report(ref_feat[feat_cols], cur_feat[feat_cols], feature_report_path, "feature_drift")
                summaries.append(summary)
                logger.info("Monitoring: feature_drift — drift=%s share=%.2f", summary["dataset_drift"], summary["share_of_drifted_columns"])
            else:
                logger.warning("Monitoring: no monitored feature columns found — skipping feature report")
        else:
            logger.warning("Monitoring: feature paths missing — skipping feature drift")

        # --- Prediction drift ---
        cur_pred_path = self.eval_artifact.scored_players_path
        ref_pred_path = ref_run_dir / "model_evaluation" / "scored_players.parquet"

        if cur_pred_path.exists() and ref_pred_path.exists():
            cur_pred = _sample(pd.read_parquet(cur_pred_path), n)
            ref_pred = _sample(pd.read_parquet(ref_pred_path), n)
            score_cols = [c for c in [_SCORE_COL] if c in cur_pred.columns and c in ref_pred.columns]
            if not score_cols:
                score_cols = [c for c in cur_pred.columns if c in ref_pred.columns and "score" in c.lower()]
            if score_cols:
                summary = _run_report(ref_pred[score_cols], cur_pred[score_cols], prediction_report_path, "prediction_drift")
                summaries.append(summary)
                logger.info("Monitoring: prediction_drift — drift=%s share=%.2f", summary["dataset_drift"], summary["share_of_drifted_columns"])
            else:
                logger.warning("Monitoring: no score columns found — skipping prediction drift")
        else:
            logger.warning("Monitoring: scored_players paths missing — skipping prediction drift")

        if not summaries:
            logger.warning("Monitoring: no reports generated — reference artifacts unavailable")
            return _EMPTY

        overall_drift = any(s["dataset_drift"] for s in summaries)
        avg_share = sum(s["share_of_drifted_columns"] for s in summaries) / len(summaries)

        drift_summary = {
            "reference_run_dir": str(ref_run_dir),
            "overall_drift_detected": overall_drift,
            "average_drift_share": round(avg_share, 4),
            "drift_threshold": threshold,
            "above_threshold": avg_share >= threshold,
            "reports": summaries,
        }
        summary_path = reports_dir / "drift_summary.json"
        write_json(drift_summary, summary_path)
        logger.info(
            "Monitoring: complete — overall_drift=%s avg_share=%.2f above_threshold=%s",
            overall_drift, avg_share, drift_summary["above_threshold"],
        )

        return MonitoringArtifact(
            reports_dir=reports_dir,
            data_drift_report_path=data_report_path if data_report_path.exists() else None,
            feature_drift_report_path=feature_report_path if feature_report_path.exists() else None,
            prediction_drift_report_path=prediction_report_path if prediction_report_path.exists() else None,
            drift_summary_path=summary_path,
            monitoring_completed=True,
        )
