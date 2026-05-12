from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from fraud_detection.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, read_json, write_json

logger = get_logger(__name__)


def _capture_stats(scores: pd.Series, labels: pd.Series, *, k: int) -> dict:
    n = len(scores)
    k_eff = max(1, min(int(k), n)) if n else 0
    total_pos = int(labels.sum()) if n else 0
    if k_eff == 0:
        return {"k": 0, "captured_fraud": 0, "capture_rate": 0.0, "precision": 0.0, "lift": 0.0}
    top_idx = scores.nlargest(k_eff).index
    hits = int(labels.loc[top_idx].sum())
    precision = hits / k_eff
    base_rate = total_pos / n if n else 0.0
    return {
        "k": k_eff,
        "captured_fraud": hits,
        "capture_rate": hits / total_pos if total_pos else 0.0,
        "precision": precision,
        "lift": precision / base_rate if base_rate else 0.0,
    }


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, training_artifact: ModelTrainingArtifact):
        self.config = config
        self.training_artifact = training_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("PartnershipModelEvaluation: starting")
        try:
            ensure_dir(self.config.output_dir)
            if self.training_artifact.stage1_oof_predictions_path is None:
                raise ValueError("stage1_oof_predictions_path is required.")
            predictions_path = self.training_artifact.stage1_oof_predictions_path.parent / "stage2_predictions.parquet"
            features_path = self.training_artifact.stage1_oof_predictions_path.parent / "stage2_features.parquet"
            predictions = pd.read_parquet(predictions_path)
            features = pd.read_parquet(features_path)
            scored = features.merge(predictions, on="member_id", how="left")
            scored["stage2_score"] = pd.to_numeric(scored.get("stage2_score", 0.0), errors="coerce").fillna(0.0)
            labels = scored["label_gold_member"].astype(int) if "label_gold_member" in scored.columns else pd.Series(0, index=scored.index)
            n = len(scored)
            top5_k = max(1, int(n * 0.05)) if n else 0
            fraud_members = int(labels.sum())
            label_status = "available" if fraud_members > 0 else "unavailable"
            capture_stats = {
                "top_5pct": _capture_stats(scored["stage2_score"], labels, k=top5_k),
                "top_50": _capture_stats(scored["stage2_score"], labels, k=50),
                "top_100": _capture_stats(scored["stage2_score"], labels, k=100),
                "top_250": _capture_stats(scored["stage2_score"], labels, k=250),
            }
            top5 = capture_stats["top_5pct"]
            gate_passed = bool(
                label_status == "unavailable"
                or (
                    float(top5["capture_rate"]) >= float(self.config.min_capture_rate_top_5pct)
                    and float(top5["lift"]) >= float(self.config.min_lift_top_5pct)
                )
            )

            scored_path = self.config.output_dir / "stage2_holdout_predictions.parquet"
            capture_table_path = self.config.output_dir / "capture_table.csv"
            report_path = self.config.output_dir / "evaluation_report.json"
            scored.to_parquet(scored_path, index=False)
            pd.DataFrame([{"bucket": key, **value} for key, value in capture_stats.items()]).to_csv(capture_table_path, index=False)
            training_report = read_json(self.training_artifact.training_report_path)
            report = {
                "model_version": "partnership_v1",
                "total_members": int(n),
                "fraud_members": fraud_members,
                "label_status": label_status,
                "gate_reason": (
                    "no_analyst_labels_available_promote_latest_artifacts"
                    if label_status == "unavailable"
                    else "label_metrics_gate"
                ),
                "stage2_capture_top_5pct": float(top5["capture_rate"]),
                "stage2_lift_top_5pct": float(top5["lift"]),
                "capture_stats": capture_stats,
                "gate_passed": gate_passed,
                "gate_thresholds": {
                    "min_capture_rate_top_5pct": self.config.min_capture_rate_top_5pct,
                    "min_lift_top_5pct": self.config.min_lift_top_5pct,
                },
                "training_report": training_report,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(report, report_path)
            return ModelEvaluationArtifact(
                stage2_holdout_predictions_path=scored_path,
                capture_rate_table_path=capture_table_path,
                evaluation_report_path=report_path,
                gate_passed=gate_passed,
                stage2_capture_rate_top_5pct=float(top5["capture_rate"]),
                stage2_lift_top_5pct=float(top5["lift"]),
                stage2_top_50_captured=int(capture_stats["top_50"]["captured_fraud"]),
            )
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc
