from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelTrainingConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, load_parquet, save_joblib, write_json

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "draws_played", "sessions_played", "active_days", "total_staked",
    "avg_stake_per_draw", "median_stake_per_draw", "stake_std", "max_stake_per_draw",
    "min_stake_per_draw", "avg_inter_draw_seconds", "std_inter_draw_seconds",
    "median_inter_draw_seconds", "min_inter_draw_seconds", "avg_nonzero_bets_per_draw",
    "median_nonzero_bets_per_draw", "avg_max_bet_share", "median_max_bet_share",
    "avg_bet_amount_std_in_draw", "avg_bet_amount_mean_in_draw", "avg_entropy",
    "entropy_std", "avg_gini", "gini_std", "avg_tiny_bet_ratio", "avg_position_coverage",
    "unique_templates", "avg_net_result", "median_net_result", "std_net_result",
    "total_net_result", "positive_draw_rate", "stake_cv", "template_reuse_ratio",
    "pnl_volatility", "win_rate", "draws_per_active_day", "avg_draws_per_session",
    "max_template_reuse", "ccs_player_count", "ccs_total_staked", "ccs_avg_bet",
]

STYLE_COLUMNS = [
    "draws_played", "avg_stake_per_draw", "avg_nonzero_bets_per_draw", "avg_max_bet_share",
    "avg_entropy", "avg_gini", "avg_tiny_bet_ratio", "avg_position_coverage",
    "template_reuse_ratio", "max_template_reuse", "stake_cv", "avg_inter_draw_seconds",
    "positive_draw_rate",
]

STYLE_LOG1P_COLUMNS = [
    "draws_played", "avg_stake_per_draw", "max_template_reuse", "avg_inter_draw_seconds",
]

ANOMALY_COMPONENT_WEIGHTS = {
    "iso_forest_score_norm": 0.40,
    "mahalanobis_norm": 0.30,
    "cluster_distance_norm": 0.30,
}


def make_model_frame(
    player_features: pd.DataFrame,
    log1p_cols: list[str],
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    active_feature_columns = feature_columns or FEATURE_COLUMNS
    available = [c for c in active_feature_columns if c in player_features.columns]
    frame = player_features[available].copy()
    for col in log1p_cols:
        if col in frame.columns:
            frame[col] = np.log1p(frame[col].clip(lower=0))
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


def make_style_frame(
    player_features: pd.DataFrame,
    style_log1p_cols: list[str],
    style_columns: list[str] | None = None,
) -> pd.DataFrame:
    active_style_columns = style_columns or STYLE_COLUMNS
    available = [c for c in active_style_columns if c in player_features.columns]
    frame = player_features[available].copy()
    for col in style_log1p_cols:
        if col in frame.columns:
            frame[col] = np.log1p(frame[col].clip(lower=0))
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, fe_artifact: FeatureEngineeringArtifact):
        self.config = config
        self.fe_artifact = fe_artifact

    def initiate_model_training(self) -> ModelTrainingArtifact:
        logger.info("ModelTraining: starting")
        try:
            ensure_dir(self.config.output_dir)
            player_df = load_parquet(self.fe_artifact.player_features_path)

            if "event_fraud_flag" not in player_df.columns:
                raise FraudDetectionException("event_fraud_flag column missing — run in training_eval mode", sys)

            log1p_cols = self.config.iso_forest_params.get("_log1p_cols", [])
            X_raw = make_model_frame(player_df, log1p_cols)
            feature_cols = list(X_raw.columns)
            y = player_df["event_fraud_flag"].astype(int).to_numpy()

            logger.info("ModelTraining: %d players, %d features, %d fraud", len(player_df), len(feature_cols), int(y.sum()))

            # --- Anomaly branch (full cohort, no train/test split) ---
            scaler_unsup = StandardScaler()
            X_scaled = scaler_unsup.fit_transform(X_raw)

            iso_params = {k: v for k, v in self.config.iso_forest_params.items() if not k.startswith("_")}
            iso_forest = IsolationForest(**iso_params)
            iso_forest.fit(X_scaled)

            mean_vec = X_scaled.mean(axis=0)
            cov_mat = np.cov(X_scaled, rowvar=False)
            cov_inv = np.linalg.pinv(cov_mat)
            mahal_stats = {"mean_vec": mean_vec, "cov_inv": cov_inv}

            kmeans = KMeans(**self.config.kmeans_params)
            kmeans.fit(X_scaled)

            style_frame = make_style_frame(player_df, STYLE_LOG1P_COLUMNS)
            style_scaler = StandardScaler()
            style_scaled = style_scaler.fit_transform(style_frame)
            style_pca = PCA(n_components=2, random_state=self.config.random_seed)
            style_pca.fit(style_scaled)

            full_pca = PCA(n_components=2, random_state=self.config.random_seed)
            full_pca.fit(X_scaled)

            # --- Supervised evaluation branch (dev/test split + OOF) ---
            all_idx = np.arange(len(player_df))
            idx_dev, idx_test, y_dev, y_test = train_test_split(
                all_idx, y,
                test_size=0.30,
                random_state=self.config.random_seed,
                stratify=y if y.sum() >= 2 else None,
            )

            lr_template = LogisticRegression(**self.config.lr_params)
            n_splits = max(2, min(5, int(y_dev.sum())))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)

            supervised_scores_eval = np.full(len(player_df), np.nan, dtype=float)
            for train_rel, val_rel in skf.split(X_raw.iloc[idx_dev], y_dev):
                fold_train_idx = idx_dev[train_rel]
                fold_val_idx = idx_dev[val_rel]
                fold_scaler = StandardScaler()
                X_fold_train = fold_scaler.fit_transform(X_raw.iloc[fold_train_idx])
                X_fold_val = fold_scaler.transform(X_raw.iloc[fold_val_idx])
                fold_lr = clone(lr_template)
                fold_lr.fit(X_fold_train, y[fold_train_idx])
                supervised_scores_eval[fold_val_idx] = fold_lr.predict_proba(X_fold_val)[:, 1]

            dev_scaler = StandardScaler()
            X_dev_scaled = dev_scaler.fit_transform(X_raw.iloc[idx_dev])
            X_test_scaled = dev_scaler.transform(X_raw.iloc[idx_test])
            lr_dev = clone(lr_template)
            lr_dev.fit(X_dev_scaled, y_dev)
            supervised_scores_eval[idx_test] = lr_dev.predict_proba(X_test_scaled)[:, 1]

            # --- Operational supervised model (full labeled cohort) ---
            scaler_operational = StandardScaler()
            X_all_scaled = scaler_operational.fit_transform(X_raw)
            lr_operational = clone(lr_template)
            lr_operational.fit(X_all_scaled, y)

            # --- Compute eval metrics ---
            from sklearn.metrics import average_precision_score, roc_auc_score
            valid_eval_mask = ~np.isnan(supervised_scores_eval)
            pr_auc = float(average_precision_score(y[valid_eval_mask], supervised_scores_eval[valid_eval_mask]))
            roc_auc = float(roc_auc_score(y[valid_eval_mask], supervised_scores_eval[valid_eval_mask]))

            # Save models
            save_joblib(iso_forest, self.config.output_dir / "iso_forest.joblib")
            save_joblib(kmeans, self.config.output_dir / "kmeans.joblib")
            save_joblib(mahal_stats, self.config.output_dir / "mahalanobis_stats.joblib")
            save_joblib(
                {
                    "scaler_unsup": scaler_unsup,
                    "scaler_operational": scaler_operational,
                    "dev_scaler": dev_scaler,
                    "style_scaler": style_scaler,
                    "style_pca": style_pca,
                    "full_pca": full_pca,
                },
                self.config.output_dir / "scaler.joblib",
            )
            save_joblib(
                {"lr_operational": lr_operational, "lr_dev": lr_dev},
                self.config.output_dir / "logistic_regression.joblib",
            )

            # Save supervised_scores_eval back to player_features for evaluation component
            player_df["supervised_score_eval"] = supervised_scores_eval
            save_joblib(player_df, self.config.output_dir / "player_df_with_eval_scores.joblib")

            report = {
                "total_players": len(player_df),
                "fraud_players": int(y.sum()),
                "dev_size": int(len(idx_dev)),
                "test_size": int(len(idx_test)),
                "n_features": len(feature_cols),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "iso_forest_params": iso_params,
                "kmeans_params": self.config.kmeans_params,
                "lr_params": self.config.lr_params,
                "anomaly_weight": self.config.anomaly_weight,
                "supervised_weight": self.config.supervised_weight,
                "anomaly_component_weights": ANOMALY_COMPONENT_WEIGHTS,
                "feature_columns": feature_cols,
                "log1p_cols": log1p_cols,
                "style_columns": STYLE_COLUMNS,
                "style_log1p_cols": STYLE_LOG1P_COLUMNS,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            report_path = self.config.output_dir / "training_report.json"
            write_json(report, report_path)

            logger.info(
                "ModelTraining: complete — PR-AUC=%.4f, ROC-AUC=%.4f",
                pr_auc, roc_auc,
            )
            return ModelTrainingArtifact(
                iso_forest_path=self.config.output_dir / "iso_forest.joblib",
                kmeans_path=self.config.output_dir / "kmeans.joblib",
                mahalanobis_stats_path=self.config.output_dir / "mahalanobis_stats.joblib",
                scaler_path=self.config.output_dir / "scaler.joblib",
                lr_operational_path=self.config.output_dir / "logistic_regression.joblib",
                training_report_path=report_path,
                feature_columns=feature_cols,
            )
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
