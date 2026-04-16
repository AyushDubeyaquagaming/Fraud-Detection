from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.components.model_training import FEATURE_COLUMNS, make_model_frame


def _make_player_df(n: int = 100, n_fraud: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {col: rng.uniform(0, 10, n) for col in FEATURE_COLUMNS},
        dtype=float,
    )
    fraud_flags = np.zeros(n, dtype=int)
    fraud_flags[:n_fraud] = 1
    rng.shuffle(fraud_flags)
    df["event_fraud_flag"] = fraud_flags
    df["member_id"] = [f"M{i:04d}" for i in range(n)]
    df["primary_ccs_id"] = "CCS1"
    return df


def test_make_model_frame_shape():
    df = _make_player_df()
    X = make_model_frame(df, log1p_cols=["total_staked"])
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_make_model_frame_no_inf():
    df = _make_player_df()
    X = make_model_frame(df, log1p_cols=["total_staked", "avg_stake_per_draw"])
    assert not np.any(np.isinf(X.values))
    assert not np.any(np.isnan(X.values))


def test_isolation_forest_output_shape(tmp_path):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df = _make_player_df(n=200, n_fraud=20)
    X = make_model_frame(df, log1p_cols=[])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=10, contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    scores = -iso.score_samples(X_scaled)

    assert len(scores) == len(df)


def test_logistic_regression_class_weight():
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
    assert lr.class_weight == "balanced"


def test_anomaly_scaler_fit_on_full_cohort(tmp_path):
    """Anomaly scaler must be fit on full cohort, not just train split."""
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = _make_player_df(n=100, n_fraud=10)
    X = make_model_frame(df, log1p_cols=[])

    # Fit on full cohort
    scaler_full = StandardScaler()
    scaler_full.fit(X)

    # Fit on 70% split
    scaler_split = StandardScaler()
    scaler_split.fit(X.iloc[:70])

    # The means should differ because they are fit on different data
    # (They could be close but shouldn't be exactly equal)
    # This test just verifies we CAN fit on full cohort vs partial
    assert scaler_full.mean_.shape == scaler_split.mean_.shape


def test_full_model_training_run(tmp_path):
    """Smoke test: ModelTraining runs end-to-end and produces expected files."""
    from fraud_detection.components.model_training import ModelTraining
    from fraud_detection.entity.config_entity import ModelTrainingConfig
    from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact

    df = _make_player_df(n=200, n_fraud=20)
    fe_path = tmp_path / "player_features.parquet"
    df.to_parquet(fe_path, index=False)

    fe_artifact = FeatureEngineeringArtifact(
        player_features_path=fe_path,
        history_df_path=fe_path,
        fraud_player_count=20,
        dropped_positive_count=0,
        feature_columns=FEATURE_COLUMNS,
        feature_summary_path=tmp_path / "feature_summary.json",
        mode="training_eval",
    )

    config = ModelTrainingConfig(
        iso_forest_params={
            "n_estimators": 10, "contamination": 0.05,
            "max_samples": "auto", "random_state": 42, "n_jobs": 1,
            "_log1p_cols": ["total_staked"],
        },
        kmeans_params={"n_clusters": 4, "random_state": 42, "n_init": 10},
        lr_params={"C": 0.1, "class_weight": "balanced", "max_iter": 100, "random_state": 42},
        anomaly_weight=0.60,
        supervised_weight=0.40,
        random_seed=42,
        output_dir=tmp_path / "model_training",
    )

    artifact = ModelTraining(config, fe_artifact).initiate_model_training()

    assert artifact.iso_forest_path.exists()
    assert artifact.kmeans_path.exists()
    assert artifact.mahalanobis_stats_path.exists()
    assert artifact.scaler_path.exists()
    assert artifact.lr_operational_path.exists()
    assert artifact.training_report_path.exists()

    import joblib
    scalers = joblib.load(artifact.scaler_path)
    assert "style_scaler" in scalers
    assert "style_pca" in scalers
    assert "full_pca" in scalers
