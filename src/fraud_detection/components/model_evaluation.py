from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fraud_detection.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, load_joblib, save_parquet, write_json

logger = get_logger(__name__)

COHORT_SCOPE_NOTE = (
    "Scores are relative to the analysis cohort (~1,045 players), "
    "not the full BetBlitz platform."
)


def _normalize_component(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.5))


def _capture_count(scores: pd.Series, labels: pd.Series, pct: float) -> int:
    threshold = scores.quantile(1 - pct)
    return int(scores[labels == 1].ge(threshold).sum())


def _save_feature_importance_plot(feature_importance_df: pd.DataFrame, output_path: Path) -> None:
    top_df = feature_importance_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Absolute coefficient")
    plt.ylabel("Feature")
    plt.title("Operational Logistic Regression Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_confusion_matrix_plot(labels: np.ndarray, predictions: np.ndarray, output_path: Path) -> dict[str, int]:
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-fraud", "Fraud"]).plot(ax=ax, colorbar=False)
    ax.set_title("Out-of-sample Supervised Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _save_correlation_heatmap_plot(player_df: pd.DataFrame, feature_names: list[str], output_path: Path) -> None:
    corr_columns = [name for name in feature_names if name in player_df.columns][:12]
    if "event_fraud_flag" in player_df.columns:
        corr_columns.append("event_fraud_flag")
    corr_df = player_df[corr_columns].corr().round(2)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Top-feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, training_artifact: ModelTrainingArtifact):
        self.config = config
        self.training_artifact = training_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("ModelEvaluation: starting")
        try:
            ensure_dir(self.config.output_dir)

            iso_forest = load_joblib(self.training_artifact.iso_forest_path)
            kmeans = load_joblib(self.training_artifact.kmeans_path)
            mahal_stats = load_joblib(self.training_artifact.mahalanobis_stats_path)
            scalers = load_joblib(self.training_artifact.scaler_path)
            lr_models = load_joblib(self.training_artifact.lr_operational_path)

            scaler_unsup = scalers["scaler_unsup"]
            scaler_operational = scalers["scaler_operational"]
            style_scaler = scalers.get("style_scaler")
            style_pca = scalers.get("style_pca")
            full_pca = scalers.get("full_pca")
            lr_operational = lr_models["lr_operational"]
            mean_vec = mahal_stats["mean_vec"]
            cov_inv = mahal_stats["cov_inv"]

            # Load player_df with eval scores saved by model_training
            player_df = load_joblib(
                self.training_artifact.iso_forest_path.parent / "player_df_with_eval_scores.joblib"
            )

            from fraud_detection.components.model_training import (
                ANOMALY_COMPONENT_WEIGHTS,
                make_model_frame,
                make_style_frame,
            )
            training_report_path = self.training_artifact.training_report_path
            import json
            with open(training_report_path) as f:
                training_report = json.load(f)

            log1p_cols = training_report.get("log1p_cols", [])
            style_columns = training_report.get("style_columns", [])
            style_log1p_cols = training_report.get("style_log1p_cols", [])
            component_weights = training_report.get("anomaly_component_weights", ANOMALY_COMPONENT_WEIGHTS)
            X_raw = make_model_frame(player_df, log1p_cols, self.training_artifact.feature_columns)

            X_unsup = scaler_unsup.transform(X_raw)

            # Anomaly scores
            iso_raw = -iso_forest.score_samples(X_unsup)
            mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_unsup])
            cluster_ids = kmeans.labels_
            cluster_raw = np.array([
                np.linalg.norm(X_unsup[i] - kmeans.cluster_centers_[cluster_ids[i]])
                for i in range(len(X_unsup))
            ])

            player_df["iso_forest_score"] = iso_raw
            player_df["iso_forest_score_norm"] = [
                _normalize_component(v, float(iso_raw.min()), float(iso_raw.max())) for v in iso_raw
            ]
            player_df["mahalanobis_dist"] = mahal_raw
            player_df["mahalanobis_norm"] = [
                _normalize_component(v, float(mahal_raw.min()), float(mahal_raw.max())) for v in mahal_raw
            ]
            player_df["cluster_id"] = cluster_ids
            player_df["cluster_distance"] = cluster_raw
            player_df["cluster_distance_norm"] = [
                _normalize_component(v, float(cluster_raw.min()), float(cluster_raw.max())) for v in cluster_raw
            ]
            player_df["anomaly_score"] = (
                float(component_weights["iso_forest_score_norm"]) * player_df["iso_forest_score_norm"]
                + float(component_weights["mahalanobis_norm"]) * player_df["mahalanobis_norm"]
                + float(component_weights["cluster_distance_norm"]) * player_df["cluster_distance_norm"]
            )

            if style_scaler is not None and style_pca is not None:
                style_frame = make_style_frame(player_df, style_log1p_cols, style_columns)
                style_coords = style_pca.transform(style_scaler.transform(style_frame))
                player_df["style_pc1"] = style_coords[:, 0]
                player_df["style_pc2"] = style_coords[:, 1]
            if full_pca is not None:
                full_coords = full_pca.transform(X_unsup)
                player_df["pc1"] = full_coords[:, 0]
                player_df["pc2"] = full_coords[:, 1]

            # Operational supervised scores
            X_operational = scaler_operational.transform(X_raw)
            player_df["supervised_score"] = lr_operational.predict_proba(X_operational)[:, 1]

            # Risk scores
            player_df["risk_score"] = (
                self.config.risk_tier_p80 * player_df["anomaly_score"]  # wrong — use weights
            )
            # Correct: use anomaly_weight / supervised_weight from training config
            # Read from training_report
            anomaly_w = training_report.get("anomaly_weight", 0.60)
            supervised_w = training_report.get("supervised_weight", 0.40)
            player_df["risk_score"] = anomaly_w * player_df["anomaly_score"] + supervised_w * player_df["supervised_score"]
            player_df["risk_score_eval"] = (
                anomaly_w * player_df["anomaly_score"] + supervised_w * player_df["supervised_score_eval"]
            )

            # Risk tiers
            p80 = float(player_df["risk_score"].quantile(self.config.risk_tier_p80))
            p95 = float(player_df["risk_score"].quantile(self.config.risk_tier_p95))
            player_df["risk_tier"] = pd.cut(
                player_df["risk_score"],
                bins=[-0.001, p80, p95, float(player_df["risk_score"].max()) + 0.001],
                labels=["LOW", "MEDIUM", "HIGH"],
            )
            player_df["risk_rank"] = player_df["risk_score"].rank(ascending=False, method="min").astype(int)
            player_df["risk_percentile"] = player_df["risk_score"].rank(pct=True)
            player_df["anomaly_percentile"] = player_df["anomaly_score"].rank(pct=True)
            player_df["supervised_percentile"] = player_df["supervised_score"].rank(pct=True)
            player_df["source"] = "precomputed"

            labels = player_df["event_fraud_flag"].astype(int)
            percentiles = self.config.threshold_percentiles
            capture_rates = {
                "anomaly": {f"top_{int(p*100)}pct": _capture_count(player_df["anomaly_score"], labels, p) for p in percentiles},
                "supervised_oos": {f"top_{int(p*100)}pct": _capture_count(player_df["supervised_score_eval"].fillna(0), labels, p) for p in percentiles},
                "combined_oos": {f"top_{int(p*100)}pct": _capture_count(player_df["risk_score_eval"].fillna(0), labels, p) for p in percentiles},
            }

            valid_eval_mask = player_df["supervised_score_eval"].notna()
            eval_predictions = (player_df.loc[valid_eval_mask, "supervised_score_eval"] >= 0.50).astype(int).to_numpy()
            eval_labels = labels.loc[valid_eval_mask].to_numpy()

            combined_oos_top_20 = capture_rates["combined_oos"]["top_20pct"]
            gate_passed = combined_oos_top_20 >= self.config.min_capture_top_20pct

            if not gate_passed:
                logger.error(
                    "Promotion gate FAILED: combined_oos_top_20pct=%d < threshold=%d",
                    combined_oos_top_20, self.config.min_capture_top_20pct,
                )
            else:
                logger.info("Promotion gate PASSED: combined_oos_top_20pct=%d", combined_oos_top_20)

            # Alert queue (top 50)
            alert_cols = [
                "member_id", "primary_ccs_id", "risk_score", "risk_tier",
                "anomaly_score", "supervised_score", "draws_played",
                "total_staked", "avg_entropy", "template_reuse_ratio", "avg_tiny_bet_ratio",
            ]
            available_alert_cols = [c for c in alert_cols if c in player_df.columns]
            alert_queue = (
                player_df.sort_values("risk_score", ascending=False)[available_alert_cols]
                .head(50)
                .reset_index(drop=True)
            )

            # Save outputs
            scored_path = self.config.output_dir / "scored_players.parquet"
            capture_path = self.config.output_dir / "capture_rate_table.csv"
            eval_path = self.config.output_dir / "evaluation_report.json"
            plots_dir = self.config.output_dir / "plots"
            ensure_dir(plots_dir)

            save_parquet(player_df, scored_path)

            # Capture rate as flat table
            capture_rows = []
            for score_type, vals in capture_rates.items():
                for pct_key, count in vals.items():
                    capture_rows.append({"score_type": score_type, "percentile": pct_key, "captured_fraud": count})
            pd.DataFrame(capture_rows).to_csv(capture_path, index=False)

            feature_importance_df = pd.DataFrame(
                {
                    "feature": self.training_artifact.feature_columns,
                    "importance": np.abs(lr_operational.coef_[0]),
                }
            ).sort_values("importance", ascending=False)
            feature_importance_csv_path = plots_dir / "feature_importance.csv"
            feature_importance_plot_path = plots_dir / "feature_importance.png"
            feature_importance_df.to_csv(feature_importance_csv_path, index=False)
            _save_feature_importance_plot(feature_importance_df, feature_importance_plot_path)

            confusion_matrix_plot_path = plots_dir / "confusion_matrix.png"
            confusion_matrix_counts = _save_confusion_matrix_plot(eval_labels, eval_predictions, confusion_matrix_plot_path)

            correlation_heatmap_path = plots_dir / "correlation_heatmap.png"
            _save_correlation_heatmap_plot(
                player_df,
                feature_importance_df["feature"].tolist(),
                correlation_heatmap_path,
            )

            evaluation = {
                "total_players": int(len(player_df)),
                "fraud_players": int(labels.sum()),
                "anomaly_weight": anomaly_w,
                "supervised_weight": supervised_w,
                "anomaly_component_weights": component_weights,
                "capture_rates": capture_rates,
                "combined_oos_top_20pct": combined_oos_top_20,
                "gate_passed": gate_passed,
                "min_capture_top_20pct": self.config.min_capture_top_20pct,
                "confusion_matrix": confusion_matrix_counts,
                "risk_tier_distribution": {
                    k: int(v) for k, v in player_df["risk_tier"].value_counts().sort_index().to_dict().items()
                },
                "risk_p80": p80,
                "risk_p95": p95,
                "plot_artifacts": {
                    "feature_importance_csv": str(feature_importance_csv_path),
                    "feature_importance_plot": str(feature_importance_plot_path),
                    "confusion_matrix_plot": str(confusion_matrix_plot_path),
                    "correlation_heatmap_plot": str(correlation_heatmap_path),
                },
                "cohort_scope_note": COHORT_SCOPE_NOTE,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(evaluation, eval_path)

            # Save alert queue to output_dir
            alert_queue.to_csv(self.config.output_dir / "alert_queue.csv", index=False)

            logger.info(
                "ModelEvaluation: complete — %d players scored, gate_passed=%s",
                len(player_df), gate_passed,
            )
            return ModelEvaluationArtifact(
                scored_players_path=scored_path,
                capture_rate_table_path=capture_path,
                evaluation_report_path=eval_path,
                combined_oos_top_20pct=combined_oos_top_20,
                gate_passed=gate_passed,
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
