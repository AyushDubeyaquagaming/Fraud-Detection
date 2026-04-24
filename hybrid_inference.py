from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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


WEEKLY_LOOKBACK_DAYS = 7
MIN_WEEKLY_HISTORY_ROWS = 5
MIN_WEEKLY_DRAWS = 5
DEFAULT_ALERT_QUEUE_SIZE = 50
WEEKLY_COHORT_CACHE_TTL_SECONDS = 900


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


def build_member_history_query(member_id: str, lookback_days: int | None = None) -> dict[str, Any]:
    normalized = validate_member_id(member_id)
    query: dict[str, Any] = {
        "member_id": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}
    }
    if lookback_days is not None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(lookback_days))
        query["trans_date"] = {"$gte": start_dt, "$lt": end_dt}
    return query


def build_date_window_query(lookback_days: int) -> dict[str, Any]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    return {"trans_date": {"$gte": start_dt, "$lt": end_dt}}


def fetch_member_history(member_id: str, lookback_days: int | None = None) -> pd.DataFrame:
    normalized = validate_member_id(member_id)
    client, collection = get_mongo_collection()
    try:
        docs = list(collection.find(build_member_history_query(normalized, lookback_days=lookback_days), MONGO_PROJECTION))
    finally:
        client.close()
    if not docs:
        if lookback_days is not None:
            raise MemberNotFoundError(
                f"{normalized} was not found in the last {int(lookback_days)} days of the MongoDB roulette source."
            )
        raise MemberNotFoundError(f"{normalized} was not found in the MongoDB roulette source.")
    return pd.DataFrame(docs)


def fetch_cohort_history(lookback_days: int) -> pd.DataFrame:
    client, collection = get_mongo_collection()
    try:
        docs = list(collection.find(build_date_window_query(lookback_days), MONGO_PROJECTION))
    finally:
        client.close()
    if not docs:
        raise InsufficientHistoryError(
            f"No MongoDB roulette activity was found in the last {int(lookback_days)} days."
        )
    return pd.DataFrame(docs)


def enforce_minimum_history(
    player_features: pd.DataFrame,
    raw_rows: int,
    min_rows: int,
    min_draws: int,
    lookback_days: int,
) -> None:
    draws_played = int(player_features.loc[0, "draws_played"])
    if raw_rows < min_rows or draws_played < min_draws:
        raise InsufficientHistoryError(
            f"Insufficient recent history for weekly scoring: last {lookback_days} days produced "
            f"{raw_rows} raw rows and {draws_played} draws; need at least {min_rows} rows and {min_draws} draws."
        )


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


def _weekly_score_notes(row: dict[str, Any], lookback_days: int) -> list[str]:
    notes = [
        f"Score computed from the last {lookback_days} days of member activity.",
        "Risk tier and rank are relative to the current weekly cohort and are not an absolute fraud probability.",
        "Evaluation-only fields are not available for weekly live scoring.",
    ]
    if int(row.get("matched_fraud_rows", 0)) > 0:
        notes.append("Pre-fraud cutoff was applied because this member matches existing fraud labels.")
    return notes


def _score_result_from_weekly_row(row: pd.Series, lookback_days: int) -> ScoreResult:
    row_dict = row.to_dict()
    reliability = str(row_dict.get("score_reliability", "higher"))
    return ScoreResult(
        scored_row=row_dict,
        source=str(row_dict.get("source", "mongo_weekly")),
        raw_rows=int(row_dict.get("raw_history_rows", 0)),
        history_rows_used=int(row_dict.get("history_rows_used", 0)),
        matched_fraud_rows=int(row_dict.get("matched_fraud_rows", 0)),
        reliability=reliability,
        notes=_weekly_score_notes(row_dict, lookback_days),
    )


def _current_cache_bucket(ttl_seconds: int = WEEKLY_COHORT_CACHE_TTL_SECONDS) -> int:
    return int(datetime.now(timezone.utc).timestamp() // int(ttl_seconds))


def _get_mahalanobis_stats(artifacts: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "mahal_stats" in artifacts:
        return artifacts["mahal_stats"]["mean_vec"], artifacts["mahal_stats"]["cov_inv"]
    return artifacts["mean_vec"], artifacts["cov_inv"]


def score_feature_frame(
    player_features: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    lookback_days: int | None = None,
    source: str,
    risk_p80: float | None = None,
    risk_p95: float | None = None,
) -> pd.DataFrame:
    scored = player_features.copy()

    model_frame = make_model_frame(
        scored,
        feature_columns=artifacts.get("feature_columns"),
        log1p_columns=artifacts.get("log1p_columns"),
    )
    style_frame = make_style_frame(
        scored,
        style_columns=artifacts.get("style_columns"),
        style_log1p_columns=artifacts.get("style_log1p_columns"),
    )

    X_unsup = artifacts["scaler_unsup"].transform(model_frame)
    iso_raw = -artifacts["iso_forest"].score_samples(X_unsup)
    mean_vec, cov_inv = _get_mahalanobis_stats(artifacts)
    mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_unsup])
    cluster_ids = artifacts["kmeans"].predict(X_unsup)
    cluster_raw = np.array([
        np.linalg.norm(X_unsup[i] - artifacts["kmeans"].cluster_centers_[cluster_ids[i]])
        for i in range(len(X_unsup))
    ])

    scored["cluster_id"] = cluster_ids
    scored["cluster_distance"] = cluster_raw
    scored["iso_forest_score"] = iso_raw
    scored["mahalanobis_dist"] = mahal_raw
    scored["iso_forest_score_norm"] = [
        normalize_component(v, artifacts["iso_min"], artifacts["iso_max"]) for v in iso_raw
    ]
    scored["mahalanobis_norm"] = [
        normalize_component(v, artifacts["mahal_min"], artifacts["mahal_max"]) for v in mahal_raw
    ]
    scored["cluster_distance_norm"] = [
        normalize_component(v, artifacts["cluster_min"], artifacts["cluster_max"]) for v in cluster_raw
    ]

    component_weights = artifacts.get(
        "anomaly_component_weights",
        {"iso_forest_score_norm": 0.40, "mahalanobis_norm": 0.30, "cluster_distance_norm": 0.30},
    )
    scored["anomaly_score"] = (
        float(component_weights["iso_forest_score_norm"]) * scored["iso_forest_score_norm"]
        + float(component_weights["mahalanobis_norm"]) * scored["mahalanobis_norm"]
        + float(component_weights["cluster_distance_norm"]) * scored["cluster_distance_norm"]
    )

    X_operational = artifacts["scaler_operational"].transform(model_frame)
    scored["supervised_score"] = artifacts["lr_operational"].predict_proba(X_operational)[:, 1]
    anomaly_weight = float(artifacts.get("anomaly_weight", ANOMALY_WEIGHT))
    supervised_weight = float(artifacts.get("supervised_weight", SUPERVISED_WEIGHT))
    scored["risk_score"] = anomaly_weight * scored["anomaly_score"] + supervised_weight * scored["supervised_score"]

    if artifacts.get("full_pca") is not None:
        full_coords = artifacts["full_pca"].transform(X_unsup)
        scored["pc1"] = full_coords[:, 0]
        scored["pc2"] = full_coords[:, 1]

    if artifacts.get("style_pca") is not None and artifacts.get("style_scaler") is not None and len(style_frame.columns) > 0:
        style_coords = artifacts["style_pca"].transform(artifacts["style_scaler"].transform(style_frame))
        scored["style_pc1"] = style_coords[:, 0]
        scored["style_pc2"] = style_coords[:, 1]

    local_p80 = float(risk_p80) if risk_p80 is not None else float(scored["risk_score"].quantile(0.80))
    local_p95 = float(risk_p95) if risk_p95 is not None else float(scored["risk_score"].quantile(0.95))
    scored["risk_tier"] = pd.cut(
        scored["risk_score"],
        bins=[-0.001, local_p80, local_p95, float(scored["risk_score"].max()) + 0.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    scored["risk_rank"] = scored["risk_score"].rank(ascending=False, method="min").astype(int)
    scored["risk_percentile"] = scored["risk_score"].rank(pct=True)
    scored["anomaly_percentile"] = scored["anomaly_score"].rank(pct=True)
    scored["supervised_percentile"] = scored["supervised_score"].rank(pct=True)
    scored["source"] = source
    scored["lookback_days"] = int(lookback_days) if lookback_days is not None else None
    return scored


def clear_weekly_scored_cohort_cache() -> None:
    _cached_weekly_scored_cohort.cache_clear()


@lru_cache(maxsize=4)
def _cached_weekly_scored_cohort(
    lookback_days: int = WEEKLY_LOOKBACK_DAYS,
    alert_queue_size: int = DEFAULT_ALERT_QUEUE_SIZE,
    cache_bucket: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    artifacts = ensure_artifacts()
    fraud_csv = load_fraud_labels()

    raw_history = fetch_cohort_history(lookback_days)
    normalized_history = normalize_member_history(raw_history)

    fraud_event_keys = set(fraud_csv["fraud_event_key"])
    event_labels = (
        normalized_history.assign(event_fraud_flag=normalized_history["fraud_event_key"].isin(fraud_event_keys).astype(int))
        .groupby("member_id", as_index=False)["event_fraud_flag"]
        .max()
    )

    usable_history, _ = apply_pre_fraud_cutoff(normalized_history, fraud_csv)
    player_features = aggregate_member_features(usable_history)

    raw_counts = normalized_history.groupby("member_id").size().reset_index(name="raw_history_rows")
    used_counts = usable_history.groupby("member_id").size().reset_index(name="history_rows_used")
    matched_fraud_rows = (
        normalized_history.assign(event_fraud_flag=normalized_history["fraud_event_key"].isin(fraud_event_keys).astype(int))
        .groupby("member_id", as_index=False)["event_fraud_flag"]
        .sum()
        .rename(columns={"event_fraud_flag": "matched_fraud_rows"})
    )

    scored = (
        player_features
        .merge(event_labels, on="member_id", how="left")
        .merge(raw_counts, on="member_id", how="left")
        .merge(used_counts, on="member_id", how="left")
        .merge(matched_fraud_rows, on="member_id", how="left")
    )
    for column in ["event_fraud_flag", "raw_history_rows", "history_rows_used", "matched_fraud_rows"]:
        scored[column] = scored[column].fillna(0).astype(int)

    scored["score_reliability"] = [
        reliability_label(draws_played=int(draws), raw_rows=int(rows))[0]
        for draws, rows in zip(scored["draws_played"], scored["raw_history_rows"])
    ]

    eligibility_mask = (
        (scored["raw_history_rows"] >= MIN_WEEKLY_HISTORY_ROWS)
        & (scored["draws_played"] >= MIN_WEEKLY_DRAWS)
    )
    scored = scored.loc[eligibility_mask].reset_index(drop=True)
    if scored.empty:
        raise InsufficientHistoryError(
            f"No members met the minimum weekly history thresholds in the last {int(lookback_days)} days."
        )

    scored = score_feature_frame(
        scored,
        artifacts,
        lookback_days=lookback_days,
        source="mongo_weekly",
    )

    queue_columns = [
        "member_id", "primary_ccs_id", "risk_score", "risk_tier", "anomaly_score", "supervised_score",
        "draws_played", "total_staked", "avg_entropy", "template_reuse_ratio",
    ]
    queue_columns = [column for column in queue_columns if column in scored.columns]
    alert_queue = (
        scored.sort_values("risk_score", ascending=False)[queue_columns]
        .head(int(alert_queue_size))
        .reset_index(drop=True)
    )

    evaluation = {
        "mode": "weekly_live",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": int(lookback_days),
        "total_players": int(len(scored)),
        "fraud_players": int(scored.get("event_fraud_flag", pd.Series(dtype=int)).sum()) if "event_fraud_flag" in scored.columns else 0,
        "alert_queue_size": int(alert_queue_size),
        "risk_tier_distribution": {tier: int((scored["risk_tier"] == tier).sum()) for tier in ["LOW", "MEDIUM", "HIGH"]},
        "score_distribution": {
            "mean": float(scored["risk_score"].mean()),
            "median": float(scored["risk_score"].median()),
            "p95": float(scored["risk_score"].quantile(0.95)),
        },
        "cohort_scope_note": f"Weekly operational cohort scored from the last {int(lookback_days)} days using the promoted model bundle.",
    }
    return scored, alert_queue, evaluation


def load_weekly_scored_cohort(
    lookback_days: int = WEEKLY_LOOKBACK_DAYS,
    alert_queue_size: int = DEFAULT_ALERT_QUEUE_SIZE,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if force_refresh:
        clear_weekly_scored_cohort_cache()
    cache_bucket = _current_cache_bucket()
    scored, alert_queue, evaluation = _cached_weekly_scored_cohort(
        int(lookback_days),
        int(alert_queue_size),
        cache_bucket,
    )
    return scored.copy(), alert_queue.copy(), deepcopy(evaluation)


def get_weekly_member_score(
    member_id: str,
    *,
    lookback_days: int = WEEKLY_LOOKBACK_DAYS,
    force_refresh: bool = False,
) -> ScoreResult:
    normalized = validate_member_id(member_id)
    weekly_scored, _, _ = load_weekly_scored_cohort(
        lookback_days=lookback_days,
        force_refresh=force_refresh,
    )
    match_df = weekly_scored.loc[weekly_scored["member_id"] == normalized]
    if not match_df.empty:
        return _score_result_from_weekly_row(match_df.iloc[0], lookback_days)

    artifacts = ensure_artifacts()
    fraud_csv = load_fraud_labels()
    raw_history = fetch_member_history(normalized, lookback_days=lookback_days)
    normalized_history = normalize_member_history(raw_history)
    usable_history, matched_fraud_rows = apply_pre_fraud_cutoff(normalized_history, fraud_csv)
    player_features = aggregate_member_features(usable_history)
    if len(player_features) != 1:
        raise HybridInferenceError("Expected exactly one aggregated player row for weekly member scoring.")

    enforce_minimum_history(
        player_features=player_features,
        raw_rows=int(len(raw_history)),
        min_rows=MIN_WEEKLY_HISTORY_ROWS,
        min_draws=MIN_WEEKLY_DRAWS,
        lookback_days=lookback_days,
    )

    scored_member = score_feature_frame(
        player_features,
        artifacts,
        lookback_days=lookback_days,
        source="mongo_live",
        risk_p80=float(weekly_scored["risk_score"].quantile(0.80)),
        risk_p95=float(weekly_scored["risk_score"].quantile(0.95)),
    )
    row = scored_member.iloc[0].to_dict()
    row.update(
        {
            "event_fraud_flag": int(matched_fraud_rows > 0),
            "raw_history_rows": int(len(raw_history)),
            "history_rows_used": int(len(usable_history)),
            "matched_fraud_rows": int(matched_fraud_rows),
            "lookback_days": int(lookback_days),
            "score_reliability": reliability_label(
                draws_played=int(player_features.loc[0, "draws_played"]),
                raw_rows=int(len(raw_history)),
            )[0],
        }
    )
    row = append_reference_rank_fields(weekly_scored, row)
    return ScoreResult(
        scored_row=row,
        source="mongo_live",
        raw_rows=int(len(raw_history)),
        history_rows_used=int(len(usable_history)),
        matched_fraud_rows=int(matched_fraud_rows),
        reliability=str(row["score_reliability"]),
        notes=_weekly_score_notes(row, lookback_days),
    )


def score_member_id(
    member_id: str,
    force_rebuild_artifacts: bool = False,
    lookback_days: int | None = None,
    require_minimum_history: bool = False,
) -> ScoreResult:
    normalized = validate_member_id(member_id)
    artifacts = ensure_artifacts(force_rebuild=force_rebuild_artifacts)
    fraud_csv = load_fraud_labels()
    reference = load_reference_scored()

    raw_history = fetch_member_history(normalized, lookback_days=lookback_days)
    normalized_history = normalize_member_history(raw_history)
    usable_history, matched_fraud_rows = apply_pre_fraud_cutoff(normalized_history, fraud_csv)
    player_features = aggregate_member_features(usable_history)
    if len(player_features) != 1:
        raise HybridInferenceError("Expected exactly one aggregated player row for live scoring.")
    if require_minimum_history and lookback_days is not None:
        enforce_minimum_history(
            player_features=player_features,
            raw_rows=int(len(raw_history)),
            min_rows=MIN_WEEKLY_HISTORY_ROWS,
            min_draws=MIN_WEEKLY_DRAWS,
            lookback_days=lookback_days,
        )

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
    if lookback_days is not None:
        notes.append(f"Score computed from the last {lookback_days} days of member activity.")
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
            "lookback_days": int(lookback_days) if lookback_days is not None else None,
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