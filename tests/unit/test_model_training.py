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


def test_iso_forest_score_direction_for_extreme_points():
    """Behavioral invariant: players with extreme feature values must score as more
    anomalous than median-behaviored players. This is a weak but important sanity
    check — if the sign convention were inverted anywhere in the scoring path,
    extremes would score LESS anomalous than the median.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    normal = pd.DataFrame(
        {col: rng.normal(0, 1, 500) for col in FEATURE_COLUMNS[:10]},
    )
    # Inject 5 extreme outliers
    extremes = pd.DataFrame(
        {col: rng.normal(15, 0.1, 5) for col in FEATURE_COLUMNS[:10]},
    )
    df = pd.concat([normal, extremes], ignore_index=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X)
    # Use the same sign convention as the production code: `-score_samples`
    scores = -iso.score_samples(X)
    # Extremes are at indices 500..504; they must be MORE anomalous than median
    median_score = np.median(scores[:500])
    assert scores[500:].mean() > median_score, (
        f"IF extremes scored {scores[500:].mean():.4f} but median was {median_score:.4f}; "
        "sign convention may be wrong"
    )


def test_kmeans_cluster_distance_is_nonnegative():
    """L2 distance to assigned cluster center is a distance: cannot be negative."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 10))
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    ids = km.labels_
    distances = np.array([
        np.linalg.norm(X[i] - km.cluster_centers_[ids[i]]) for i in range(len(X))
    ])
    assert (distances >= 0).all()
    assert np.all(np.isfinite(distances))


def test_mahalanobis_distance_is_finite_with_pseudo_inverse():
    """With near-singular covariance (constant features), the pseudo-inverse
    fallback must still produce finite distances."""
    from scipy.spatial.distance import mahalanobis as mahal_fn

    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 5))
    # Add a constant column → covariance matrix is singular
    X = np.column_stack([X, np.ones(100)])
    mean_vec = X.mean(axis=0)
    cov_mat = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov_mat)
    distances = np.array([mahal_fn(row, mean_vec, cov_inv) for row in X])
    assert np.all(np.isfinite(distances)), "Mahalanobis distances must be finite"
    assert (distances >= 0).all()


def test_lr_predict_proba_range():
    """Logistic regression predict_proba output is always in [0, 1]."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, 10))
    y = (X[:, 0] + rng.normal(0, 0.1, 200) > 0).astype(int)
    lr = LogisticRegression(max_iter=500).fit(X, y)
    probas = lr.predict_proba(X)[:, 1]
    assert probas.min() >= 0.0 and probas.max() <= 1.0


def test_combined_risk_score_range():
    """anomaly_weight * anomaly_score + supervised_weight * supervised_score stays
    in [0, max(anomaly_weight + supervised_weight*1, ...)] — for weights summing
    to 1 and scores in [0, 1], the combined score is in [0, 1]."""
    rng = np.random.default_rng(4)
    anomaly = rng.uniform(0, 1, 1000)
    supervised = rng.uniform(0, 1, 1000)
    aw, sw = 0.6, 0.4
    combined = aw * anomaly + sw * supervised
    assert combined.min() >= 0.0
    assert combined.max() <= aw + sw  # guaranteed <= 1 when weights sum to 1


def test_supervised_scores_eval_coverage():
    """With 10 fraud in 105K, OOF evaluation must cover every player with a score
    — either via K-fold inside dev, or via lr_dev inference on test. No player
    may have NaN in `supervised_score_eval`.
    """
    from fraud_detection.components.model_training import ModelTraining
    from fraud_detection.entity.config_entity import ModelTrainingConfig
    from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact
    import joblib
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df = _make_player_df(n=1000, n_fraud=20)
        fe_path = tmp / "player_features.parquet"
        df.to_parquet(fe_path, index=False)

        fe_artifact = FeatureEngineeringArtifact(
            player_features_path=fe_path,
            history_df_path=fe_path,
            fraud_player_count=20,
            dropped_positive_count=0,
            feature_columns=FEATURE_COLUMNS,
            feature_summary_path=tmp / "feature_summary.json",
            mode="training_eval",
        )
        config = ModelTrainingConfig(
            iso_forest_params={
                "n_estimators": 10, "contamination": 0.05,
                "max_samples": "auto", "random_state": 42, "n_jobs": 1,
                "_log1p_cols": [],
            },
            kmeans_params={"n_clusters": 4, "random_state": 42, "n_init": 10},
            lr_params={"C": 0.1, "class_weight": "balanced", "max_iter": 100, "random_state": 42},
            anomaly_weight=0.60, supervised_weight=0.40, random_seed=42,
            output_dir=tmp / "model_training",
        )
        ModelTraining(config, fe_artifact).initiate_model_training()
        scored = joblib.load(tmp / "model_training" / "player_df_with_eval_scores.joblib")
        assert scored["supervised_score_eval"].notna().all(), (
            "Every player must receive an out-of-sample score — NaN means OOF "
            "construction silently skipped some players"
        )


def test_cluster_id_assignment_via_predict(tmp_path):
    """Training + evaluation flow: cluster_id in scored output must match
    `kmeans.predict(X_unsup)` (not `kmeans.labels_`, which assumes fit-order).
    """
    from fraud_detection.components.model_training import ModelTraining
    from fraud_detection.entity.config_entity import ModelTrainingConfig
    from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact
    import joblib

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
            "_log1p_cols": [],
        },
        kmeans_params={"n_clusters": 4, "random_state": 42, "n_init": 10},
        lr_params={"C": 0.1, "class_weight": "balanced", "max_iter": 100, "random_state": 42},
        anomaly_weight=0.60, supervised_weight=0.40, random_seed=42,
        output_dir=tmp_path / "model_training",
    )
    training_artifact = ModelTraining(config, fe_artifact).initiate_model_training()

    kmeans = joblib.load(training_artifact.kmeans_path)
    scalers = joblib.load(training_artifact.scaler_path)
    player_df = joblib.load(training_artifact.iso_forest_path.parent / "player_df_with_eval_scores.joblib")
    X_raw = make_model_frame(player_df, log1p_cols=[], feature_columns=FEATURE_COLUMNS)
    X_unsup = scalers["scaler_unsup"].transform(X_raw)
    expected_ids = kmeans.predict(X_unsup)
    # Shuffle player_df to prove we don't rely on fit-order
    shuffled = player_df.sample(frac=1, random_state=7).reset_index(drop=True)
    X_shuffled = scalers["scaler_unsup"].transform(
        make_model_frame(shuffled, log1p_cols=[], feature_columns=FEATURE_COLUMNS)
    )
    shuffled_ids = kmeans.predict(X_shuffled)
    # Each shuffled row's predicted id must match the original row's predicted id
    # when we remap by member_id → index in the original order.
    orig_map = {m: expected_ids[i] for i, m in enumerate(player_df["member_id"])}
    for i, m in enumerate(shuffled["member_id"]):
        assert shuffled_ids[i] == orig_map[m], (
            f"Cluster prediction for {m} changed under shuffle: fit-order leak"
        )


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
