from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_cache"
CURRENT_DIR = ROOT / "artifacts" / "current"

# Prefer artifacts/current/ (MLOps pipeline output); fall back to data_cache/ legacy paths
def _resolve_scored_path() -> Path:
    new_path = CURRENT_DIR / "hybrid_scored_players.parquet"
    legacy_path = DATA_DIR / "hybrid_scored_players.parquet"
    return new_path if new_path.exists() else legacy_path

def _resolve_artifact_path() -> Path:
    new_path = CURRENT_DIR / "model_bundle.joblib"
    legacy_path = DATA_DIR / "hybrid_inference_artifacts.joblib"
    return new_path if new_path.exists() else legacy_path

SCORED_PATH = _resolve_scored_path()
FRAUD_CSV = ROOT / "ROULET CHEATING DATA.csv"
ENV_PATH = ROOT / ".env"
ARTIFACT_PATH = _resolve_artifact_path()

RANDOM_SEED = 42
ANOMALY_WEIGHT = 0.60
SUPERVISED_WEIGHT = 0.40

FEATURE_COLUMNS = [
    "draws_played",
    "sessions_played",
    "active_days",
    "total_staked",
    "avg_stake_per_draw",
    "median_stake_per_draw",
    "stake_std",
    "max_stake_per_draw",
    "min_stake_per_draw",
    "avg_inter_draw_seconds",
    "std_inter_draw_seconds",
    "median_inter_draw_seconds",
    "min_inter_draw_seconds",
    "avg_nonzero_bets_per_draw",
    "median_nonzero_bets_per_draw",
    "avg_max_bet_share",
    "median_max_bet_share",
    "avg_bet_amount_std_in_draw",
    "avg_bet_amount_mean_in_draw",
    "avg_entropy",
    "entropy_std",
    "avg_gini",
    "gini_std",
    "avg_tiny_bet_ratio",
    "avg_position_coverage",
    "unique_templates",
    "avg_net_result",
    "median_net_result",
    "std_net_result",
    "total_net_result",
    "positive_draw_rate",
    "stake_cv",
    "template_reuse_ratio",
    "pnl_volatility",
    "win_rate",
    "draws_per_active_day",
    "avg_draws_per_session",
    "max_template_reuse",
    "ccs_player_count",
    "ccs_total_staked",
    "ccs_avg_bet",
]

STYLE_COLUMNS = [
    "draws_played",
    "avg_stake_per_draw",
    "avg_nonzero_bets_per_draw",
    "avg_max_bet_share",
    "avg_entropy",
    "avg_gini",
    "avg_tiny_bet_ratio",
    "avg_position_coverage",
    "template_reuse_ratio",
    "max_template_reuse",
    "stake_cv",
    "avg_inter_draw_seconds",
    "positive_draw_rate",
]

LOG1P_COLUMNS = {
    "total_staked",
    "avg_stake_per_draw",
    "median_stake_per_draw",
    "stake_std",
    "max_stake_per_draw",
    "ccs_total_staked",
    "ccs_avg_bet",
    "stake_cv",
    "avg_bet_amount_std_in_draw",
    "avg_bet_amount_mean_in_draw",
    "std_net_result",
    "pnl_volatility",
}

STYLE_LOG1P_COLUMNS = {
    "draws_played",
    "avg_stake_per_draw",
    "max_template_reuse",
    "avg_inter_draw_seconds",
}

MONGO_PROJECTION = {
    "member_id": 1,
    "draw_id": 1,
    "bets": 1,
    "win_points": 1,
    "total_bet_amount": 1,
    "session_id": 1,
    "ccs_id": 1,
    "createdAt": 1,
    "updatedAt": 1,
    "trans_date": 1,
}

VALID_MEMBER_ID = re.compile(r"^[A-Z0-9_-]{4,64}$")


class HybridInferenceError(Exception):
    pass


class InvalidMemberIdError(HybridInferenceError):
    pass


class MemberNotFoundError(HybridInferenceError):
    pass


class InsufficientHistoryError(HybridInferenceError):
    pass


@dataclass
class ScoreResult:
    scored_row: dict[str, Any]
    source: str
    raw_rows: int
    history_rows_used: int
    matched_fraud_rows: int
    reliability: str
    notes: list[str]


LIVE_OUTPUT_EXCLUDE_FIELDS = {"supervised_score_eval", "risk_score_eval"}


def validate_member_id(member_id: str) -> str:
    normalized = str(member_id).strip().upper()
    if not VALID_MEMBER_ID.match(normalized):
        raise InvalidMemberIdError("Member ID must be 4-64 chars and contain only A-Z, 0-9, _ or -.")
    return normalized


def parse_bets(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


def safe_entropy(amounts: list[float]) -> float:
    values = np.array([amount for amount in amounts if amount > 0], dtype=float)
    if values.sum() == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(scipy_entropy(probabilities, base=2))


def gini_coeff(amounts: list[float]) -> float:
    values = np.sort(np.array([amount for amount in amounts if amount > 0], dtype=float))
    count = len(values)
    if count == 0 or values.sum() == 0:
        return 0.0
    idx = np.arange(1, count + 1)
    return float((2 * np.sum(idx * values) / (count * values.sum())) - (count + 1) / count)


def make_bet_template(bets_list: Any) -> tuple:
    if not isinstance(bets_list, list):
        return tuple()
    return tuple(
        sorted(
            (
                str(bet.get("number", "")),
                round(float(bet.get("bet_amount", 0) or 0), 6),
            )
            for bet in bets_list
            if float(bet.get("bet_amount", 0) or 0) > 0
        )
    )


def compute_draw_features(bets_list: list[dict[str, Any]]) -> dict[str, float]:
    amounts = [float(bet.get("bet_amount", 0) or 0) for bet in bets_list]
    nonzero = [amount for amount in amounts if amount > 0]
    total_amount = sum(amounts)
    max_amount = max(amounts) if amounts else 0.0
    nonzero_count = len(nonzero)
    return {
        "bets_per_draw": len(amounts),
        "nonzero_bets_per_draw": nonzero_count,
        "tiny_bet_ratio_in_draw": sum(1 for amount in nonzero if amount <= 1) / max(nonzero_count, 1),
        "max_bet_share_in_draw": max_amount / total_amount if total_amount > 0 else 0.0,
        "bet_amount_std_in_draw": float(np.std(nonzero)) if nonzero_count > 1 else 0.0,
        "bet_amount_mean_in_draw": float(np.mean(nonzero)) if nonzero_count > 0 else 0.0,
        "entropy_in_draw": safe_entropy(amounts),
        "gini_in_draw": max(gini_coeff(amounts), 0.0),
        "unique_positions_in_draw": nonzero_count,
        "position_coverage": nonzero_count / 38.0,
    }


def mode_val(series: pd.Series):
    modes = series.mode()
    return modes.iloc[0] if len(modes) else np.nan


def coerce_datetime_value(value: Any):
    if isinstance(value, dict) and "$date" in value:
        return value["$date"]
    return value


def normalize_timestamp(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "createdAt.$date",
        "createdat.$date",
        "trans_date.$date",
        "updatedAt.$date",
        "ts",
        "createdAt",
        "trans_date",
        "updatedAt",
    ]
    timestamp = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for column in candidates:
        if column not in df.columns:
            continue
        series = pd.to_datetime(df[column].map(coerce_datetime_value), utc=True, errors="coerce")
        timestamp = timestamp.fillna(series)
    return timestamp


def load_fraud_labels() -> pd.DataFrame:
    fraud_csv = pd.read_csv(FRAUD_CSV)
    fraud_csv.columns = [column.strip().lower() for column in fraud_csv.columns]
    fraud_csv["member_id_norm"] = fraud_csv["member_id"].astype(str).str.strip().str.upper()
    fraud_csv["draw_id_norm"] = pd.to_numeric(fraud_csv["draw_id"], errors="coerce").astype("Int64")
    fraud_csv["fraud_event_key"] = fraud_csv["draw_id_norm"].astype(str) + "|" + fraud_csv["member_id_norm"]
    return fraud_csv


def normalize_member_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()
    df = raw_df.copy()
    df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
    df["draw_id"] = pd.to_numeric(df["draw_id"], errors="coerce").astype("Int64")
    df["ts"] = normalize_timestamp(df)
    df["bets_parsed"] = df["bets"].apply(parse_bets)

    draw_features = pd.DataFrame(df["bets_parsed"].apply(compute_draw_features).tolist())
    df = pd.concat([df.reset_index(drop=True), draw_features.reset_index(drop=True)], axis=1)
    df["win_points"] = pd.to_numeric(df.get("win_points", 0), errors="coerce").fillna(0.0)
    df["total_bet_amount"] = pd.to_numeric(df.get("total_bet_amount", 0), errors="coerce").fillna(0.0)
    df["session_id"] = pd.to_numeric(df.get("session_id", 0), errors="coerce").fillna(0).astype(int)
    df["ccs_id"] = df.get("ccs_id", "").astype(str)
    df["net_result"] = df["win_points"] - df["total_bet_amount"]
    df["bet_template"] = df["bets_parsed"].apply(make_bet_template)
    df = df.sort_values(["member_id", "ts", "draw_id"])
    df["inter_draw_seconds"] = df.groupby("member_id")["ts"].diff().dt.total_seconds()
    df["fraud_event_key"] = df["draw_id"].astype(str) + "|" + df["member_id"]
    return df


def apply_pre_fraud_cutoff(member_df: pd.DataFrame, fraud_csv: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    fraud_event_keys = set(fraud_csv["fraud_event_key"])
    df = member_df.copy()
    df["event_label"] = df["fraud_event_key"].isin(fraud_event_keys).astype(int)

    event_match_df = df.loc[df["event_label"] == 1, ["member_id", "draw_id", "ts"]].copy()
    if event_match_df.empty:
        return df.copy(), 0

    first_fraud = (
        event_match_df.sort_values(["member_id", "ts", "draw_id"])
        .groupby("member_id", as_index=False)
        .agg(
            first_fraud_ts=("ts", "min"),
            first_fraud_draw_id=("draw_id", "min"),
            matched_fraud_rows=("draw_id", "size"),
        )
    )
    df = df.merge(
        first_fraud[["member_id", "first_fraud_ts", "first_fraud_draw_id"]].assign(is_fraud_player=1),
        on="member_id",
        how="left",
    )
    df["is_fraud_player"] = df["is_fraud_player"].fillna(0).astype(int)
    pre_fraud_mask = df["is_fraud_player"].eq(1) & (
        (df["ts"].notna() & df["first_fraud_ts"].notna() & (df["ts"] < df["first_fraud_ts"]))
        | (df["first_fraud_ts"].isna() & df["first_fraud_draw_id"].notna() & (df["draw_id"] < df["first_fraud_draw_id"]))
    )
    non_fraud_mask = df["is_fraud_player"].eq(0)
    history_df = df.loc[pre_fraud_mask | non_fraud_mask].copy()
    return history_df, int(df["event_label"].sum())


def aggregate_member_features(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        raise InsufficientHistoryError("No usable history remains after applying the pre-fraud cutoff.")

    player_agg = history_df.groupby("member_id").agg(
        draws_played=("draw_id", "nunique"),
        sessions_played=("session_id", "nunique"),
        active_days=("ts", lambda x: x.dt.date.nunique()),
        total_staked=("total_bet_amount", "sum"),
        avg_stake_per_draw=("total_bet_amount", "mean"),
        median_stake_per_draw=("total_bet_amount", "median"),
        stake_std=("total_bet_amount", "std"),
        max_stake_per_draw=("total_bet_amount", "max"),
        min_stake_per_draw=("total_bet_amount", "min"),
        avg_inter_draw_seconds=("inter_draw_seconds", "mean"),
        std_inter_draw_seconds=("inter_draw_seconds", "std"),
        median_inter_draw_seconds=("inter_draw_seconds", "median"),
        min_inter_draw_seconds=("inter_draw_seconds", "min"),
        avg_nonzero_bets_per_draw=("nonzero_bets_per_draw", "mean"),
        median_nonzero_bets_per_draw=("nonzero_bets_per_draw", "median"),
        avg_max_bet_share=("max_bet_share_in_draw", "mean"),
        median_max_bet_share=("max_bet_share_in_draw", "median"),
        avg_bet_amount_std_in_draw=("bet_amount_std_in_draw", "mean"),
        avg_bet_amount_mean_in_draw=("bet_amount_mean_in_draw", "mean"),
        avg_entropy=("entropy_in_draw", "mean"),
        entropy_std=("entropy_in_draw", "std"),
        avg_gini=("gini_in_draw", "mean"),
        gini_std=("gini_in_draw", "std"),
        avg_tiny_bet_ratio=("tiny_bet_ratio_in_draw", "mean"),
        avg_position_coverage=("position_coverage", "mean"),
        unique_templates=("bet_template", "nunique"),
        avg_net_result=("net_result", "mean"),
        median_net_result=("net_result", "median"),
        std_net_result=("net_result", "std"),
        total_net_result=("net_result", "sum"),
        positive_draw_rate=("net_result", lambda x: (x > 0).mean()),
        primary_ccs_id=("ccs_id", mode_val),
    ).reset_index()

    for column in ["stake_std", "entropy_std", "gini_std", "std_net_result", "std_inter_draw_seconds"]:
        player_agg[column] = player_agg[column].fillna(0)
    for column in ["avg_inter_draw_seconds", "median_inter_draw_seconds", "min_inter_draw_seconds"]:
        player_agg[column] = player_agg[column].replace([np.inf, -np.inf], np.nan).fillna(0)

    player_agg["stake_cv"] = (
        player_agg["stake_std"] / player_agg["avg_stake_per_draw"].replace(0, np.nan)
    ).fillna(0)
    player_agg["template_reuse_ratio"] = (
        1 - (player_agg["unique_templates"] / player_agg["draws_played"].replace(0, np.nan))
    ).fillna(0).clip(lower=0)
    player_agg["pnl_volatility"] = (
        player_agg["std_net_result"] / player_agg["avg_stake_per_draw"].replace(0, np.nan)
    ).fillna(0)
    player_agg["win_rate"] = player_agg["positive_draw_rate"]
    player_agg["draws_per_active_day"] = (
        player_agg["draws_played"] / player_agg["active_days"].replace(0, np.nan)
    ).fillna(0)

    session_draws = (
        history_df.groupby(["member_id", "session_id"])["draw_id"]
        .nunique()
        .reset_index(name="draws_in_session")
    )
    avg_session_draws = session_draws.groupby("member_id")["draws_in_session"].mean().reset_index(name="avg_draws_per_session")
    player_agg = player_agg.merge(avg_session_draws, on="member_id", how="left")
    player_agg["avg_draws_per_session"] = player_agg["avg_draws_per_session"].fillna(1)

    max_reuse = (
        history_df.groupby("member_id")["bet_template"]
        .apply(lambda values: values.value_counts().iloc[0] if len(values) else 1)
        .reset_index(name="max_template_reuse")
    )
    player_agg = player_agg.merge(max_reuse, on="member_id", how="left")

    ccs_player_count = history_df.groupby("ccs_id")["member_id"].nunique().reset_index(name="ccs_player_count")
    ccs_totals = (
        history_df.groupby("ccs_id")
        .agg(
            ccs_total_staked=("total_bet_amount", "sum"),
            ccs_avg_bet=("total_bet_amount", "mean"),
        )
        .reset_index()
    )
    player_agg = (
        player_agg
        .merge(ccs_player_count.rename(columns={"ccs_id": "primary_ccs_id"}), on="primary_ccs_id", how="left")
        .merge(ccs_totals.rename(columns={"ccs_id": "primary_ccs_id"}), on="primary_ccs_id", how="left")
    )
    for column in ["ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]:
        player_agg[column] = player_agg[column].fillna(0)

    return player_agg


def make_model_frame(
    player_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    log1p_columns: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    active_feature_columns = feature_columns or FEATURE_COLUMNS
    active_log1p_columns = log1p_columns or LOG1P_COLUMNS
    frame = player_features[active_feature_columns].copy()
    for column in active_log1p_columns:
        if column in frame.columns:
            frame[column] = np.log1p(frame[column].clip(lower=0))
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


def make_style_frame(
    player_features: pd.DataFrame,
    style_columns: list[str] | None = None,
    style_log1p_columns: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    active_style_columns = style_columns or STYLE_COLUMNS
    active_style_log1p_columns = style_log1p_columns or STYLE_LOG1P_COLUMNS
    frame = player_features[active_style_columns].copy()
    for column in active_style_log1p_columns:
        if column in frame.columns:
            frame[column] = np.log1p(frame[column].clip(lower=0))
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


def reliability_label(draws_played: int, raw_rows: int) -> tuple[str, list[str]]:
    notes: list[str] = []
    if raw_rows < 5 or draws_played < 5:
        notes.append("Very little history is available; score stability is low.")
        return "low", notes
    if raw_rows < 20 or draws_played < 20:
        notes.append("History is limited; treat this score as directional rather than definitive.")
        return "medium", notes
    return "higher", notes


def load_reference_scored() -> pd.DataFrame:
    path = _resolve_scored_path()
    scored = pd.read_parquet(path).copy()
    scored["member_id"] = scored["member_id"].astype(str).str.upper().str.strip()
    return scored


def _is_new_bundle(artifacts: dict[str, Any]) -> bool:
    """True if artifacts came from the MLOps model_bundle (v1 pipeline), not the legacy format."""
    return "anomaly_weight" in artifacts and "mahal_stats" in artifacts


def build_artifacts(force_rebuild: bool = False) -> dict[str, Any]:
    if ARTIFACT_PATH.exists() and not force_rebuild:
        return joblib.load(ARTIFACT_PATH)

    reference = load_reference_scored()
    X_raw = make_model_frame(reference)
    X_unsup = StandardScaler().fit_transform(X_raw)

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
    cov_mat = np.cov(X_scaled, rowvar=False)
    cov_inv = np.linalg.pinv(cov_mat)
    mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_scaled])

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(X_scaled)
    cluster_raw = np.array([
        np.linalg.norm(X_scaled[index] - kmeans.cluster_centers_[kmeans.labels_[index]])
        for index in range(len(X_scaled))
    ])

    y = reference["event_fraud_flag"].astype(int).to_numpy()
    scaler_operational = StandardScaler()
    X_operational = scaler_operational.fit_transform(X_raw)
    lr_operational = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    lr_operational.fit(X_operational, y)

    style_frame = make_style_frame(reference)
    style_scaler = StandardScaler()
    style_scaled = style_scaler.fit_transform(style_frame)
    style_pca = PCA(n_components=2, random_state=RANDOM_SEED)
    style_pca.fit(style_scaled)

    full_pca = PCA(n_components=2, random_state=RANDOM_SEED)
    full_pca.fit(X_scaled)

    artifacts = {
        "feature_columns": FEATURE_COLUMNS,
        "style_columns": STYLE_COLUMNS,
        "log1p_columns": sorted(LOG1P_COLUMNS),
        "style_log1p_columns": sorted(STYLE_LOG1P_COLUMNS),
        "anomaly_component_weights": {
            "iso_forest_score_norm": 0.40,
            "mahalanobis_norm": 0.30,
            "cluster_distance_norm": 0.30,
        },
        "scaler_unsup": scaler_unsup,
        "iso_forest": iso_forest,
        "iso_min": float(iso_raw.min()),
        "iso_max": float(iso_raw.max()),
        "mean_vec": mean_vec,
        "cov_inv": cov_inv,
        "mahal_min": float(mahal_raw.min()),
        "mahal_max": float(mahal_raw.max()),
        "kmeans": kmeans,
        "cluster_min": float(cluster_raw.min()),
        "cluster_max": float(cluster_raw.max()),
        "scaler_operational": scaler_operational,
        "lr_operational": lr_operational,
        "risk_p80": float(reference["risk_score"].quantile(0.80)),
        "risk_p95": float(reference["risk_score"].quantile(0.95)),
        "full_pca": full_pca,
        "style_scaler": style_scaler,
        "style_pca": style_pca,
        "reference_size": int(len(reference)),
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    return artifacts


def ensure_artifacts(force_rebuild: bool = False) -> dict[str, Any]:
    artifact_path = _resolve_artifact_path()
    # If the new MLOps model_bundle exists and we're not forced to rebuild, use it directly
    if artifact_path == CURRENT_DIR / "model_bundle.joblib" and artifact_path.exists() and not force_rebuild:
        return joblib.load(artifact_path)
    return build_artifacts(force_rebuild=force_rebuild)


def get_mongo_collection():
    load_dotenv(ENV_PATH)
    uri = os.getenv("MONGODB_URI")
    database = os.getenv("MONGODB_DATABASE")
    collection_name = os.getenv("MONGODB_COLLECTION_ROULETTE_REPORT")
    if not uri or not database or not collection_name:
        raise HybridInferenceError("MongoDB environment variables are missing. Check .env.")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    return client, client[database][collection_name]


def fetch_member_history(member_id: str) -> pd.DataFrame:
    normalized = validate_member_id(member_id)
    client, collection = get_mongo_collection()
    try:
        docs = list(collection.find({"member_id": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}}, MONGO_PROJECTION))
    finally:
        client.close()
    if not docs:
        raise MemberNotFoundError(f"{normalized} was not found in the MongoDB roulette source.")
    return pd.DataFrame(docs)


def normalize_component(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return float(np.clip(normalized, 0.0, 1.5))


def append_reference_rank_fields(reference: pd.DataFrame, row: dict[str, Any]) -> dict[str, Any]:
    risk_scores = reference["risk_score"].astype(float)
    anomaly_scores = reference["anomaly_score"].astype(float)
    supervised_scores = reference["supervised_score"].astype(float)

    row["risk_rank"] = int((risk_scores > float(row["risk_score"])).sum() + 1)
    row["risk_percentile"] = float((risk_scores <= float(row["risk_score"])).mean())
    row["anomaly_percentile"] = float((anomaly_scores <= float(row["anomaly_score"])).mean())
    row["supervised_percentile"] = float((supervised_scores <= float(row["supervised_score"])).mean())
    return row


def score_member_id(member_id: str, force_rebuild_artifacts: bool = False) -> ScoreResult:
    normalized = validate_member_id(member_id)
    artifacts = ensure_artifacts(force_rebuild=force_rebuild_artifacts)
    fraud_csv = load_fraud_labels()
    reference = load_reference_scored()

    raw_history = fetch_member_history(normalized)
    normalized_history = normalize_member_history(raw_history)
    usable_history, matched_fraud_rows = apply_pre_fraud_cutoff(normalized_history, fraud_csv)
    player_features = aggregate_member_features(usable_history)
    if len(player_features) != 1:
        raise HybridInferenceError("Expected exactly one aggregated player row for live scoring.")

    model_frame = make_model_frame(
        player_features,
        feature_columns=artifacts.get("feature_columns"),
        log1p_columns=artifacts.get("log1p_columns"),
    )
    style_frame = make_style_frame(
        player_features,
        style_columns=artifacts.get("style_columns"),
        style_log1p_columns=artifacts.get("style_log1p_columns"),
    )

    X_unsup = artifacts["scaler_unsup"].transform(model_frame)
    iso_raw = float(-artifacts["iso_forest"].score_samples(X_unsup)[0])
    mahal_raw = float(mahalanobis(X_unsup[0], artifacts["mean_vec"], artifacts["cov_inv"]))
    cluster_id = int(artifacts["kmeans"].predict(X_unsup)[0])
    cluster_raw = float(np.linalg.norm(X_unsup[0] - artifacts["kmeans"].cluster_centers_[cluster_id]))

    iso_norm = normalize_component(iso_raw, artifacts["iso_min"], artifacts["iso_max"])
    mahal_norm = normalize_component(mahal_raw, artifacts["mahal_min"], artifacts["mahal_max"])
    cluster_norm = normalize_component(cluster_raw, artifacts["cluster_min"], artifacts["cluster_max"])
    component_weights = artifacts.get(
        "anomaly_component_weights",
        {"iso_forest_score_norm": 0.40, "mahalanobis_norm": 0.30, "cluster_distance_norm": 0.30},
    )
    anomaly_score = float(
        float(component_weights["iso_forest_score_norm"]) * iso_norm
        + float(component_weights["mahalanobis_norm"]) * mahal_norm
        + float(component_weights["cluster_distance_norm"]) * cluster_norm
    )

    X_operational = artifacts["scaler_operational"].transform(model_frame)
    supervised_score = float(artifacts["lr_operational"].predict_proba(X_operational)[0, 1])
    anomaly_weight = float(artifacts.get("anomaly_weight", ANOMALY_WEIGHT))
    supervised_weight = float(artifacts.get("supervised_weight", SUPERVISED_WEIGHT))
    risk_score = float(anomaly_weight * anomaly_score + supervised_weight * supervised_score)

    if risk_score <= artifacts["risk_p80"]:
        risk_tier = "LOW"
    elif risk_score <= artifacts["risk_p95"]:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "HIGH"

    full_coords = artifacts["full_pca"].transform(X_unsup)[0] if "full_pca" in artifacts else [np.nan, np.nan]
    if "style_pca" in artifacts and "style_scaler" in artifacts:
        style_coords = artifacts["style_pca"].transform(artifacts["style_scaler"].transform(style_frame))[0]
    else:
        style_coords = [np.nan, np.nan]
    draws_played = int(player_features.loc[0, "draws_played"])
    reliability, notes = reliability_label(draws_played=draws_played, raw_rows=len(raw_history))
    notes.append("Risk tier is relative to the current analysis cohort and is not an absolute fraud probability.")
    if matched_fraud_rows:
        notes.append("Pre-fraud cutoff was applied because this member matches existing fraud labels.")
    notes.append("Evaluation-only fields are not available for one-off live scoring.")

    row = player_features.iloc[0].to_dict()
    row.update(
        {
            "member_id": normalized,
            "event_fraud_flag": int(matched_fraud_rows > 0),
            "iso_forest_score": iso_raw,
            "iso_forest_score_norm": iso_norm,
            "mahalanobis_dist": mahal_raw,
            "mahalanobis_norm": mahal_norm,
            "cluster_id": cluster_id,
            "cluster_distance": cluster_raw,
            "cluster_distance_norm": cluster_norm,
            "anomaly_score": anomaly_score,
            "supervised_score": supervised_score,
            "supervised_score_eval": None,
            "risk_score": risk_score,
            "risk_score_eval": None,
            "risk_tier": risk_tier,
            "pc1": float(full_coords[0]),
            "pc2": float(full_coords[1]),
            "style_pc1": float(style_coords[0]),
            "style_pc2": float(style_coords[1]),
            "raw_history_rows": int(len(raw_history)),
            "history_rows_used": int(len(usable_history)),
            "matched_fraud_rows": int(matched_fraud_rows),
            "score_reliability": reliability,
            "source": "mongo_live",
        }
    )
    row = append_reference_rank_fields(reference, row)
    return ScoreResult(
        scored_row=row,
        source="mongo_live",
        raw_rows=int(len(raw_history)),
        history_rows_used=int(len(usable_history)),
        matched_fraud_rows=int(matched_fraud_rows),
        reliability=reliability,
        notes=notes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a roulette member ID with the hybrid inference pipeline.")
    parser.add_argument("member_id", help="Member ID to score from live MongoDB history")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild saved inference artifacts before scoring")
    parser.add_argument("--full-json", action="store_true", help="Include evaluation-only fields in the printed JSON output")
    args = parser.parse_args()

    result = score_member_id(args.member_id, force_rebuild_artifacts=args.rebuild)
    output_row = result.scored_row.copy()
    if not args.full_json:
        for field in LIVE_OUTPUT_EXCLUDE_FIELDS:
            output_row.pop(field, None)
    print(json.dumps(output_row, indent=2, default=str))