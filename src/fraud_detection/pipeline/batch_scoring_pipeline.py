from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from fraud_detection.constants.constants import (
    CONFIG_FILE_PATH,
    CURRENT_DIR,
    MODEL_BUNDLE_FILE,
    REPO_ROOT,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import load_joblib, read_yaml, write_json

logger = get_logger(__name__)

COHORT_SCOPE_NOTE = (
    "Scores are relative to the analysis cohort (~1,045 players), "
    "not the full BetBlitz platform."
)


def _normalize_component(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.5))


class BatchScoringPipeline:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config_path = config_path

    def run(self) -> Path:
        logger.info("BatchScoringPipeline: starting")
        try:
            config = read_yaml(self.config_path)
            batch_cfg = config.get("batch_scoring", {})
            mode = batch_cfg.get("mode", "operational")
            require_labels = batch_cfg.get("require_labels", False)

            logger.info("BatchScoringPipeline: mode=%s, require_labels=%s", mode, require_labels)

            # Load model bundle
            bundle_path = CURRENT_DIR / MODEL_BUNDLE_FILE
            if not bundle_path.exists():
                raise FileNotFoundError(
                    f"Model bundle not found: {bundle_path}. Run training pipeline first."
                )
            bundle = load_joblib(bundle_path)
            logger.info("Loaded model bundle from %s", bundle_path)

            # Load data
            ing_cfg = config["data_ingestion"]
            if ing_cfg["source"] == "parquet":
                raw_df = pd.read_parquet(REPO_ROOT / ing_cfg["parquet_path"])
            else:
                from fraud_detection.utils.mongodb import pull_full_collection
                raw_df = pull_full_collection(
                    uri_env_var=ing_cfg["mongodb"]["uri_env_var"],
                    db_env_var=ing_cfg["mongodb"]["database_env_var"],
                    collection_env_var=ing_cfg["mongodb"]["collection_env_var"],
                )

            logger.info("Loaded %d raw rows", len(raw_df))

            # Feature engineering
            from fraud_detection.entity.config_entity import FeatureEngineeringConfig
            from fraud_detection.entity.artifact_entity import DataIngestionArtifact
            from fraud_detection.components.feature_engineering import FeatureEngineering

            val_cfg = config["data_validation"]
            fe_cfg = config["feature_engineering"]

            # Save raw_df temporarily for the ingestion artifact pattern
            tmp_raw_path = CURRENT_DIR / "_tmp_scoring_raw.parquet"
            raw_df.to_parquet(tmp_raw_path, index=False)

            ingestion_artifact = DataIngestionArtifact(
                raw_data_path=tmp_raw_path,
                ingestion_report_path=CURRENT_DIR / "_tmp_ingestion_report.json",
                row_count=len(raw_df),
                member_count=int(raw_df["member_id"].nunique()) if "member_id" in raw_df.columns else 0,
                source_type=ing_cfg["source"],
            )

            if mode == "replay_eval":
                fe_mode = "training_eval"
            else:
                fe_mode = "operational"

            fe_config = FeatureEngineeringConfig(
                exclude_cols=fe_cfg["exclude_cols"],
                log1p_cols=fe_cfg["log1p_cols"],
                apply_pre_fraud_cutoff=(fe_mode == "training_eval"),
                fraud_csv_path=REPO_ROOT / val_cfg["fraud_csv_path"],
                output_dir=CURRENT_DIR / "_tmp_fe",
                mode=fe_mode,
            )
            fe_artifact = FeatureEngineering(fe_config, ingestion_artifact).initiate_feature_engineering()
            player_df = pd.read_parquet(fe_artifact.player_features_path)

            # Score with bundle
            from fraud_detection.components.model_training import (
                ANOMALY_COMPONENT_WEIGHTS,
                make_model_frame,
                make_style_frame,
            )

            feat_cfg_path = CURRENT_DIR / "feature_pipeline_config.json"
            import json
            feature_columns = bundle.get("feature_columns", [])
            if feat_cfg_path.exists():
                with open(feat_cfg_path) as f:
                    feat_cfg = json.load(f)
                log1p_cols = bundle.get("log1p_columns") or feat_cfg.get("log1p_cols", fe_cfg["log1p_cols"])
                style_columns = bundle.get("style_columns") or feat_cfg.get("style_columns", [])
                style_log1p_cols = bundle.get("style_log1p_columns") or feat_cfg.get("style_log1p_cols", [])
            else:
                log1p_cols = bundle.get("log1p_columns") or fe_cfg["log1p_cols"]
                style_columns = bundle.get("style_columns", [])
                style_log1p_cols = bundle.get("style_log1p_columns", [])

            X_raw = make_model_frame(player_df, log1p_cols, feature_columns)

            iso_forest = bundle["iso_forest"]
            kmeans = bundle["kmeans"]
            mean_vec = bundle["mahal_stats"]["mean_vec"]
            cov_inv = bundle["mahal_stats"]["cov_inv"]
            scaler_unsup = bundle["scaler_unsup"]
            scaler_operational = bundle["scaler_operational"]
            lr_operational = bundle["lr_operational"]
            anomaly_w = float(bundle.get("anomaly_weight", 0.60))
            supervised_w = float(bundle.get("supervised_weight", 0.40))
            component_weights = bundle.get("anomaly_component_weights", ANOMALY_COMPONENT_WEIGHTS)

            X_unsup = scaler_unsup.transform(X_raw)
            iso_raw = -iso_forest.score_samples(X_unsup)
            mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_unsup])
            cluster_ids = kmeans.predict(X_unsup)
            cluster_raw = np.array([
                np.linalg.norm(X_unsup[i] - kmeans.cluster_centers_[cluster_ids[i]])
                for i in range(len(X_unsup))
            ])

            player_df["cluster_id"] = cluster_ids
            player_df["cluster_distance"] = cluster_raw
            player_df["iso_forest_score_norm"] = [
                _normalize_component(v, bundle["iso_min"], bundle["iso_max"]) for v in iso_raw
            ]
            player_df["mahalanobis_norm"] = [
                _normalize_component(v, bundle["mahal_min"], bundle["mahal_max"]) for v in mahal_raw
            ]
            player_df["cluster_distance_norm"] = [
                _normalize_component(v, bundle["cluster_min"], bundle["cluster_max"]) for v in cluster_raw
            ]
            player_df["anomaly_score"] = (
                float(component_weights["iso_forest_score_norm"]) * player_df["iso_forest_score_norm"]
                + float(component_weights["mahalanobis_norm"]) * player_df["mahalanobis_norm"]
                + float(component_weights["cluster_distance_norm"]) * player_df["cluster_distance_norm"]
            )

            if bundle.get("style_scaler") is not None and bundle.get("style_pca") is not None and style_columns:
                style_frame = make_style_frame(player_df, style_log1p_cols, style_columns)
                style_coords = bundle["style_pca"].transform(bundle["style_scaler"].transform(style_frame))
                player_df["style_pc1"] = style_coords[:, 0]
                player_df["style_pc2"] = style_coords[:, 1]

            if bundle.get("full_pca") is not None:
                full_coords = bundle["full_pca"].transform(X_unsup)
                player_df["pc1"] = full_coords[:, 0]
                player_df["pc2"] = full_coords[:, 1]

            X_operational = scaler_operational.transform(X_raw)
            player_df["supervised_score"] = lr_operational.predict_proba(X_operational)[:, 1]
            player_df["risk_score"] = anomaly_w * player_df["anomaly_score"] + supervised_w * player_df["supervised_score"]

            p80 = float(bundle.get("risk_p80", player_df["risk_score"].quantile(0.80)))
            p95 = float(bundle.get("risk_p95", player_df["risk_score"].quantile(0.95)))
            player_df["risk_tier"] = pd.cut(
                player_df["risk_score"],
                bins=[-0.001, p80, p95, float(player_df["risk_score"].max()) + 0.001],
                labels=["LOW", "MEDIUM", "HIGH"],
            )
            player_df["source"] = f"batch_{mode}"

            # Save outputs
            scored_path = CURRENT_DIR / "hybrid_scored_players.parquet"
            player_df.to_parquet(scored_path, index=False)

            # Alert queue (drop label columns for operational mode)
            alert_cols = [
                "member_id", "risk_score", "risk_tier", "anomaly_score", "supervised_score",
                "draws_played", "total_staked", "avg_entropy", "template_reuse_ratio",
            ]
            if "primary_ccs_id" in player_df.columns:
                alert_cols.insert(1, "primary_ccs_id")
            alert_cols_available = [c for c in alert_cols if c in player_df.columns]
            alert_queue = (
                player_df.sort_values("risk_score", ascending=False)[alert_cols_available]
                .head(50)
                .reset_index(drop=True)
            )
            alert_path = CURRENT_DIR / "alert_queue.csv"
            alert_queue.to_csv(alert_path, index=False)

            # Evaluation summary (only if labels available)
            eval_summary: dict = {
                "mode": mode,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "total_players": len(player_df),
                "score_distribution": {
                    "mean": float(player_df["risk_score"].mean()),
                    "median": float(player_df["risk_score"].median()),
                    "p95": float(player_df["risk_score"].quantile(0.95)),
                },
                "cohort_scope_note": COHORT_SCOPE_NOTE,
            }

            if mode == "replay_eval" and "event_fraud_flag" in player_df.columns:
                labels = player_df["event_fraud_flag"].astype(int)

                def capture(scores, pct):
                    t = scores.quantile(1 - pct)
                    return int(scores[labels == 1].ge(t).sum())

                eval_summary["fraud_players"] = int(labels.sum())
                eval_summary["capture_rates"] = {
                    "top_1pct": capture(player_df["risk_score"], 0.01),
                    "top_5pct": capture(player_df["risk_score"], 0.05),
                    "top_10pct": capture(player_df["risk_score"], 0.10),
                    "top_20pct": capture(player_df["risk_score"], 0.20),
                }

            eval_path = CURRENT_DIR / "hybrid_evaluation.json"
            write_json(eval_summary, eval_path)

            # Scoring report
            report_path = CURRENT_DIR / "batch_scoring_report.json"
            write_json(
                {
                    "run_at": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                    "total_players": len(player_df),
                    "source": ing_cfg["source"],
                    "scored_path": str(scored_path),
                    "alert_path": str(alert_path),
                },
                report_path,
            )

            # Cleanup tmp files
            for tmp_f in [tmp_raw_path]:
                try:
                    tmp_f.unlink(missing_ok=True)
                except Exception:
                    pass

            logger.info(
                "BatchScoringPipeline: complete — %d players scored, alert_queue at %s",
                len(player_df), alert_path,
            )
            return CURRENT_DIR

        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
