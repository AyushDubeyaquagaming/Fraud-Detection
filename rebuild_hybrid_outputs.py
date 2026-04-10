from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from hybrid_inference import (
    ANOMALY_WEIGHT,
    ARTIFACT_PATH,
    DATA_DIR,
    RANDOM_SEED,
    SCORED_PATH,
    SUPERVISED_WEIGHT,
    aggregate_member_features,
    apply_pre_fraud_cutoff,
    build_artifacts,
    load_fraud_labels,
    make_model_frame,
    make_style_frame,
    normalize_component,
    normalize_member_history,
)


RAW_PATH = DATA_DIR / "fraud_modeling_pull.parquet"
PLAYER_FEATURE_PATH = DATA_DIR / "player_feature_table.parquet"
PLAYER_FEATURE_DICT_PATH = DATA_DIR / "player_feature_dictionary.csv"
ALERT_QUEUE_PATH = DATA_DIR / "alert_queue.csv"
HYBRID_EVAL_PATH = DATA_DIR / "hybrid_evaluation.json"


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def capture_count(scores: pd.Series, labels: pd.Series, pct: float) -> int:
    threshold = scores.quantile(1 - pct)
    return int(scores[labels == 1].ge(threshold).sum())


def build_player_table() -> tuple[pd.DataFrame, int, int]:
    raw_df = pd.read_parquet(RAW_PATH)
    fraud_df = load_fraud_labels()
    normalized_df = normalize_member_history(raw_df)
    history_df, matched_fraud_rows = apply_pre_fraud_cutoff(normalized_df, fraud_df)
    player_df = aggregate_member_features(history_df)

    fraud_event_keys = set(fraud_df["fraud_event_key"])
    fraud_players = set(normalized_df.loc[normalized_df["fraud_event_key"].isin(fraud_event_keys), "member_id"])
    player_df["event_fraud_flag"] = player_df["member_id"].isin(fraud_players).astype(int)

    dropped_positive_players = len(fraud_players) - int(player_df["event_fraud_flag"].sum())

    print(f"Raw cohort rows: {len(raw_df):,}")
    print(f"History rows after pre-fraud cutoff: {len(history_df):,}")
    print(f"Player rows: {len(player_df):,}")
    print(f"Matched fraud event rows: {matched_fraud_rows:,}")
    print(f"Usable fraud players: {int(player_df['event_fraud_flag'].sum()):,}")
    print(f"Dropped positive players with no usable history: {dropped_positive_players:,}")

    return player_df, matched_fraud_rows, dropped_positive_players


def add_unsupervised_scores(player_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], StandardScaler, KMeans, PCA, StandardScaler, PCA]:
    X_raw = make_model_frame(player_df)
    scaler_unsup = StandardScaler()
    X_scaled = scaler_unsup.fit_transform(X_raw)

    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        max_samples="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    iso_raw = -iso_forest.score_samples(X_scaled)

    mean_vec = X_scaled.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(X_scaled, rowvar=False))
    mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_scaled])

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(X_scaled)
    cluster_id = kmeans.labels_
    cluster_raw = np.array([
        np.linalg.norm(X_scaled[index] - kmeans.cluster_centers_[cluster_id[index]])
        for index in range(len(X_scaled))
    ])

    player_df["iso_forest_score"] = iso_raw
    player_df["iso_forest_score_norm"] = [normalize_component(value, float(iso_raw.min()), float(iso_raw.max())) for value in iso_raw]
    player_df["mahalanobis_dist"] = mahal_raw
    player_df["mahalanobis_norm"] = [normalize_component(value, float(mahal_raw.min()), float(mahal_raw.max())) for value in mahal_raw]
    player_df["cluster_id"] = cluster_id
    player_df["cluster_distance"] = cluster_raw
    player_df["cluster_distance_norm"] = [normalize_component(value, float(cluster_raw.min()), float(cluster_raw.max())) for value in cluster_raw]
    player_df["anomaly_score"] = (
        0.40 * player_df["iso_forest_score_norm"]
        + 0.30 * player_df["mahalanobis_norm"]
        + 0.30 * player_df["cluster_distance_norm"]
    )

    pca_full = PCA(n_components=2, random_state=RANDOM_SEED)
    full_coords = pca_full.fit_transform(X_scaled)
    player_df["pc1"] = full_coords[:, 0]
    player_df["pc2"] = full_coords[:, 1]

    style_frame = make_style_frame(player_df)
    style_scaler = StandardScaler()
    style_scaled = style_scaler.fit_transform(style_frame)
    style_pca = PCA(n_components=2, random_state=RANDOM_SEED)
    style_coords = style_pca.fit_transform(style_scaled)
    player_df["style_pc1"] = style_coords[:, 0]
    player_df["style_pc2"] = style_coords[:, 1]

    return player_df, list(X_raw.columns), scaler_unsup, kmeans, pca_full, style_scaler, style_pca


def add_supervised_scores(player_df: pd.DataFrame) -> pd.DataFrame:
    X_df = make_model_frame(player_df)
    y = player_df["event_fraud_flag"].astype(int).to_numpy()
    all_idx = np.arange(len(player_df))

    idx_dev, idx_test, y_dev, y_test = train_test_split(
        all_idx,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y if y.sum() >= 2 else None,
    )

    player_df["supervised_score_eval"] = np.nan

    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    n_splits = max(2, min(5, int(y_dev.sum())))
    oof_dev_scores = np.full(len(idx_dev), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_rel, val_rel in skf.split(X_df.iloc[idx_dev], y_dev):
        fold_train_idx = idx_dev[train_rel]
        fold_val_idx = idx_dev[val_rel]

        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(X_df.iloc[fold_train_idx])
        X_fold_val = fold_scaler.transform(X_df.iloc[fold_val_idx])

        fold_lr = clone(lr)
        fold_lr.fit(X_fold_train, y[fold_train_idx])
        oof_dev_scores[val_rel] = fold_lr.predict_proba(X_fold_val)[:, 1]

    dev_scaler = StandardScaler()
    X_dev_scaled = dev_scaler.fit_transform(X_df.iloc[idx_dev])
    X_test_scaled = dev_scaler.transform(X_df.iloc[idx_test])
    lr_dev_model = clone(lr)
    lr_dev_model.fit(X_dev_scaled, y_dev)

    player_df.loc[idx_dev, "supervised_score_eval"] = oof_dev_scores
    player_df.loc[idx_test, "supervised_score_eval"] = lr_dev_model.predict_proba(X_test_scaled)[:, 1]

    operational_scaler = StandardScaler()
    X_all_scaled = operational_scaler.fit_transform(X_df)
    lr_operational = clone(lr)
    lr_operational.fit(X_all_scaled, y)
    player_df["supervised_score"] = lr_operational.predict_proba(X_all_scaled)[:, 1]

    return player_df


def add_risk_outputs(player_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    player_df["risk_score"] = ANOMALY_WEIGHT * player_df["anomaly_score"] + SUPERVISED_WEIGHT * player_df["supervised_score"]
    player_df["risk_score_eval"] = ANOMALY_WEIGHT * player_df["anomaly_score"] + SUPERVISED_WEIGHT * player_df["supervised_score_eval"]

    p80 = float(player_df["risk_score"].quantile(0.80))
    p95 = float(player_df["risk_score"].quantile(0.95))
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

    alert_queue = (
        player_df.sort_values("risk_score", ascending=False)[
            [
                "member_id",
                "primary_ccs_id",
                "risk_score",
                "risk_tier",
                "anomaly_score",
                "supervised_score",
                "draws_played",
                "total_staked",
                "avg_entropy",
                "template_reuse_ratio",
                "avg_tiny_bet_ratio",
            ]
        ]
        .head(50)
        .reset_index(drop=True)
    )

    labels = player_df["event_fraud_flag"].astype(int)
    capture_rates = {
        "anomaly": {
            "top_1pct": capture_count(player_df["anomaly_score"], labels, 0.01),
            "top_5pct": capture_count(player_df["anomaly_score"], labels, 0.05),
            "top_10pct": capture_count(player_df["anomaly_score"], labels, 0.10),
            "top_20pct": capture_count(player_df["anomaly_score"], labels, 0.20),
        },
        "supervised_oos": {
            "top_1pct": capture_count(player_df["supervised_score_eval"], labels, 0.01),
            "top_5pct": capture_count(player_df["supervised_score_eval"], labels, 0.05),
            "top_10pct": capture_count(player_df["supervised_score_eval"], labels, 0.10),
            "top_20pct": capture_count(player_df["supervised_score_eval"], labels, 0.20),
        },
        "combined_oos": {
            "top_1pct": capture_count(player_df["risk_score_eval"], labels, 0.01),
            "top_5pct": capture_count(player_df["risk_score_eval"], labels, 0.05),
            "top_10pct": capture_count(player_df["risk_score_eval"], labels, 0.10),
            "top_20pct": capture_count(player_df["risk_score_eval"], labels, 0.20),
        },
    }

    evaluation = {
        "total_players": int(len(player_df)),
        "fraud_players": int(labels.sum()),
        "usable_fraud_players_after_cutoff": int(labels.sum()),
        "dropped_positive_players": 0,
        "anomaly_weight": ANOMALY_WEIGHT,
        "supervised_weight": SUPERVISED_WEIGHT,
        "capture_rates": capture_rates,
        "risk_tier_distribution": {key: int(value) for key, value in player_df["risk_tier"].value_counts().sort_index().to_dict().items()},
    }
    return player_df, alert_queue, evaluation


def save_feature_dictionary(player_df: pd.DataFrame) -> None:
    descriptions = {
        "draws_played": "Unique roulette draws observed before cutoff.",
        "sessions_played": "Unique sessions observed before cutoff.",
        "active_days": "Unique active days derived from timestamps.",
        "total_staked": "Total wagered amount across retained history.",
        "avg_stake_per_draw": "Mean wager per draw.",
        "median_stake_per_draw": "Median wager per draw.",
        "stake_std": "Standard deviation of wager per draw.",
        "max_stake_per_draw": "Maximum wager in a single draw.",
        "min_stake_per_draw": "Minimum wager in a single draw.",
        "avg_inter_draw_seconds": "Average seconds between consecutive draws.",
        "std_inter_draw_seconds": "Volatility of inter-draw timing.",
        "avg_nonzero_bets_per_draw": "Average count of positive-stake positions per draw.",
        "avg_max_bet_share": "Average largest-bet share within a draw.",
        "avg_entropy": "Average Shannon entropy of within-draw bet allocation.",
        "avg_gini": "Average concentration of within-draw bet allocation.",
        "avg_tiny_bet_ratio": "Average share of positive bets with stake <= 1.",
        "template_reuse_ratio": "Share of draws that repeat an existing bet template.",
        "max_template_reuse": "Largest reuse count of a single bet template.",
        "positive_draw_rate": "Fraction of draws with positive net result.",
        "draws_per_active_day": "Draw frequency per active day.",
        "ccs_player_count": "Number of unique players in the dominant CCS bucket.",
        "ccs_total_staked": "Total staked amount in the dominant CCS bucket.",
        "ccs_avg_bet": "Average bet amount in the dominant CCS bucket.",
    }
    feature_df = pd.DataFrame(
        [{"feature": column, "description": descriptions.get(column, "Hybrid feature used in notebook 03 / inference.")} for column in player_df.columns]
    )
    feature_df.to_csv(PLAYER_FEATURE_DICT_PATH, index=False)


def main() -> None:
    print_header("Rebuild Hybrid Outputs")
    player_df, matched_fraud_rows, dropped_positive_players = build_player_table()
    player_df["matched_fraud_rows_total"] = matched_fraud_rows
    base_feature_df = player_df.copy()

    print_header("Compute Unsupervised Scores")
    player_df, _, _, _, _, _, _ = add_unsupervised_scores(player_df)
    print(player_df[["anomaly_score"]].describe().round(4).to_string())

    print_header("Compute Supervised Scores")
    player_df = add_supervised_scores(player_df)
    print(player_df[["supervised_score", "supervised_score_eval"]].describe().round(4).to_string())

    print_header("Assemble Risk Outputs")
    player_df, alert_queue, evaluation = add_risk_outputs(player_df)
    evaluation["dropped_positive_players"] = dropped_positive_players

    base_feature_df.to_parquet(PLAYER_FEATURE_PATH, index=False)
    player_df.to_parquet(SCORED_PATH, index=False)
    alert_queue.to_csv(ALERT_QUEUE_PATH, index=False)
    with open(HYBRID_EVAL_PATH, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2)
    save_feature_dictionary(base_feature_df)

    if ARTIFACT_PATH.exists():
        ARTIFACT_PATH.unlink()
    build_artifacts(force_rebuild=True)

    print(f"Saved scored cohort: {SCORED_PATH}")
    print(f"Saved alert queue : {ALERT_QUEUE_PATH}")
    print(f"Saved evaluation  : {HYBRID_EVAL_PATH}")
    print(f"Rebuilt artifacts : {ARTIFACT_PATH}")
    print("Top 10 alert queue:")
    print(alert_queue.head(10).to_string(index=False))


if __name__ == "__main__":
    main()