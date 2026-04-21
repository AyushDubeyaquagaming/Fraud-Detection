from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from fraud_detection.constants.constants import (
    CONFIG_FILE_PATH,
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


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


class BatchScoringPipeline:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config_path = config_path

    def run(self) -> Path:
        logger.info("BatchScoringPipeline: starting")
        try:
            config = read_yaml(self.config_path)
            current_dir = _resolve_repo_path(config["pipeline"]["current_dir"])
            batch_cfg = config.get("batch_scoring", {})
            mode = batch_cfg.get("mode", "operational")
            require_labels = batch_cfg.get("require_labels", False)
            op_filter_cfg = batch_cfg.get("operational_filter", {}) or {}
            op_filter_enabled = bool(op_filter_cfg.get("enabled", False))
            op_min_draws = int(op_filter_cfg.get("min_draws_played", 0))
            alert_queue_size = int(batch_cfg.get("alert_queue_size", 50))

            logger.info(
                "BatchScoringPipeline: mode=%s, require_labels=%s, "
                "operational_filter=%s (min_draws_played=%d), alert_queue_size=%d, current_dir=%s",
                mode, require_labels, op_filter_enabled, op_min_draws, alert_queue_size, current_dir,
            )

            # Load model bundle
            bundle_path = current_dir / MODEL_BUNDLE_FILE
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
                from fraud_detection.utils.mongodb import (
                    build_query_batches_from_strategy,
                    pull_query_batches_to_dataframe,
                )
                mongo_cfg = ing_cfg["mongodb"]
                strategy = mongo_cfg.get("strategy", "date_window")
                strategy_params = dict(mongo_cfg.get("strategy_params", {}))
                logger.info(
                    "BatchScoringPipeline: mongodb strategy=%s, params=%s",
                    strategy,
                    strategy_params,
                )
                query_filters = build_query_batches_from_strategy(strategy, strategy_params)
                raw_df = pull_query_batches_to_dataframe(
                    uri_env_var=mongo_cfg["uri_env_var"],
                    db_env_var=mongo_cfg["database_env_var"],
                    collection_env_var=mongo_cfg["collection_env_var"],
                    query_filters=query_filters,
                )

            logger.info("Loaded %d raw rows", len(raw_df))

            # Feature engineering
            from fraud_detection.entity.config_entity import FeatureEngineeringConfig
            from fraud_detection.entity.artifact_entity import DataIngestionArtifact
            from fraud_detection.components.feature_engineering import FeatureEngineering

            val_cfg = config["data_validation"]
            fe_cfg = config["feature_engineering"]

            # Save raw_df temporarily for the ingestion artifact pattern
            current_dir.mkdir(parents=True, exist_ok=True)
            tmp_raw_path = current_dir / "_tmp_scoring_raw.parquet"
            raw_df.to_parquet(tmp_raw_path, index=False)

            ingestion_artifact = DataIngestionArtifact(
                raw_data_path=tmp_raw_path,
                ingestion_report_path=current_dir / "_tmp_ingestion_report.json",
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
                output_dir=current_dir / "_tmp_fe",
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

            feat_cfg_path = current_dir / "feature_pipeline_config.json"
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

            # Save full scored population BEFORE applying the operational filter so
            # downstream analyses can always see the un-filtered view if needed.
            scored_path = current_dir / "hybrid_scored_players.parquet"
            player_df.to_parquet(scored_path, index=False)

            # Operational pre-filter: applied at scoring time only, not during training.
            # Training still sees the full cohort so the unsupervised scaler reflects
            # the full population baseline.
            n_before_filter = len(player_df)
            n_filtered_out = 0
            if op_filter_enabled and "draws_played" in player_df.columns:
                eligible_mask = player_df["draws_played"] >= op_min_draws
                n_filtered_out = int((~eligible_mask).sum())
                alert_source = player_df.loc[eligible_mask]
                logger.info(
                    "operational_filter: kept %d / %d players (min_draws_played=%d)",
                    len(alert_source), n_before_filter, op_min_draws,
                )
            else:
                alert_source = player_df

            # Alert queue (drop label columns for operational mode)
            alert_cols = [
                "member_id", "risk_score", "risk_tier", "anomaly_score", "supervised_score",
                "draws_played", "total_staked", "avg_entropy", "template_reuse_ratio",
            ]
            if "primary_ccs_id" in player_df.columns:
                alert_cols.insert(1, "primary_ccs_id")
            alert_cols_available = [c for c in alert_cols if c in alert_source.columns]
            alert_queue = (
                alert_source.sort_values("risk_score", ascending=False)[alert_cols_available]
                .head(alert_queue_size)
                .reset_index(drop=True)
            )
            alert_path = current_dir / "alert_queue.csv"
            alert_queue.to_csv(alert_path, index=False)

            # Evaluation summary (only if labels available)
            eval_summary: dict = {
                "mode": mode,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "total_players": len(player_df),
                "operational_filter": {
                    "enabled": op_filter_enabled,
                    "min_draws_played": op_min_draws,
                    "players_before_filter": n_before_filter,
                    "players_after_filter": len(alert_source),
                    "players_filtered_out": n_filtered_out,
                },
                "alert_queue_size": alert_queue_size,
                "score_distribution": {
                    "mean": float(player_df["risk_score"].mean()),
                    "median": float(player_df["risk_score"].median()),
                    "p95": float(player_df["risk_score"].quantile(0.95)),
                },
                "cohort_scope_note": COHORT_SCOPE_NOTE,
            }

            if mode == "replay_eval" and "event_fraud_flag" in player_df.columns:
                labels = player_df["event_fraud_flag"].astype(int)
                n_total = len(player_df)
                total_fraud = int(labels.sum())
                base_rate = total_fraud / n_total if n_total > 0 else 0.0

                def stats_at_k(scores: pd.Series, k: int) -> dict:
                    k = max(1, min(k, n_total))
                    top_idx = scores.nlargest(k).index
                    captured = int(labels.loc[top_idx].sum())
                    cap_rate = captured / total_fraud if total_fraud > 0 else 0.0
                    precision = captured / k
                    lift = precision / base_rate if base_rate > 0 else 0.0
                    return {
                        "k": k,
                        "captured_fraud": captured,
                        "capture_rate": cap_rate,
                        "precision": precision,
                        "lift": lift,
                    }

                eval_summary["fraud_players"] = total_fraud
                eval_summary["base_rate"] = base_rate
                scores = player_df["risk_score"]
                eval_summary["capture_stats"] = {
                    "top_1pct": stats_at_k(scores, max(1, int(n_total * 0.01))),
                    "top_5pct": stats_at_k(scores, max(1, int(n_total * 0.05))),
                    "top_10pct": stats_at_k(scores, max(1, int(n_total * 0.10))),
                    "top_20pct": stats_at_k(scores, max(1, int(n_total * 0.20))),
                    "top_50": stats_at_k(scores, 50),
                    "top_500": stats_at_k(scores, 500),
                }
                # Legacy capture_rates (count-only) retained for backward compatibility.
                eval_summary["capture_rates"] = {
                    b: s["captured_fraud"] for b, s in eval_summary["capture_stats"].items()
                }

            eval_path = current_dir / "hybrid_evaluation.json"
            write_json(eval_summary, eval_path)

            # Scoring report
            scoring_report: dict = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "total_players": len(player_df),
                "operational_filter": {
                    "enabled": op_filter_enabled,
                    "min_draws_played": op_min_draws,
                    "players_before_filter": n_before_filter,
                    "players_after_filter": len(alert_source),
                    "players_filtered_out": n_filtered_out,
                },
                "alert_queue_size": alert_queue_size,
                "alert_queue_rows": len(alert_queue),
                "source": ing_cfg["source"],
                "scored_path": str(scored_path),
                "alert_path": str(alert_path),
            }
            if ing_cfg["source"] == "mongodb":
                scoring_report["strategy_used"] = mongo_cfg.get("strategy", "date_window")
                scoring_report["query_count"] = len(query_filters)
            report_path = current_dir / "batch_scoring_report.json"
            write_json(scoring_report, report_path)

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
            return current_dir

        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
