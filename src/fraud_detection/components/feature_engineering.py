from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import entropy as scipy_entropy

from fraud_detection.entity.artifact_entity import DataIngestionArtifact, FeatureEngineeringArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, save_parquet, write_json

logger = get_logger(__name__)

FEATURE_ENGINEERING_IN_MEMORY_MAX_ROWS = 2_000_000
FEATURE_ENGINEERING_BATCH_SIZE = 100_000
FEATURE_ENGINEERING_TARGET_ROWS_PER_BUCKET = 500_000
FEATURE_ENGINEERING_MAX_BUCKETS = 256
FEATURE_ENGINEERING_RAW_COLUMNS = [
    "member_id",
    "draw_id",
    "bets",
    "win_points",
    "total_bet_amount",
    "session_id",
    "ccs_id",
    "createdAt",
    "updatedAt",
    "trans_date",
]
TRAINING_HISTORY_COLUMNS = [
    "event_label",
    "first_fraud_ts",
    "first_fraud_draw_id",
    "is_fraud_player",
]
HISTORY_INT_COLUMNS = ["event_label", "first_fraud_draw_id"]
HISTORY_TS_COLUMNS = ["ts", "first_fraud_ts"]
TIMESTAMP_CANDIDATES = [
    "createdAt.$date",
    "createdat.$date",
    "trans_date.$date",
    "updatedAt.$date",
    "ts",
    "createdAt",
    "trans_date",
    "updatedAt",
]
COLLUSION_FEATURE_COLUMNS = [
    "max_cohort_coverage_in_draws",
    "mean_cohort_coverage_in_draws",
    "pct_draws_in_cohort_2plus",
    "mean_cohort_size",
    "mean_pairwise_jaccard_when_in_cohort",
]
ROULETTE_POSITION_COUNT = 38


# ---------------------------------------------------------------------------
# Pure helper functions (reproduced from hybrid_inference.py / notebook 03)
# ---------------------------------------------------------------------------

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
    values = np.array([a for a in amounts if a > 0], dtype=float)
    if values.sum() == 0:
        return 0.0
    probs = values / values.sum()
    return float(scipy_entropy(probs, base=2))


def gini_coeff(amounts: list[float]) -> float:
    values = np.sort(np.array([a for a in amounts if a > 0], dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * values) / (n * values.sum())) - (n + 1) / n)


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


def make_bet_template_key(bets_list: Any) -> str:
    return json.dumps(make_bet_template(bets_list), default=str, separators=(",", ":"))


def compute_draw_features(bets_list: list[dict[str, Any]]) -> dict[str, float]:
    amounts = [float(bet.get("bet_amount", 0) or 0) for bet in bets_list]
    nonzero = [a for a in amounts if a > 0]
    total = sum(amounts)
    max_amt = max(amounts) if amounts else 0.0
    nz_count = len(nonzero)
    return {
        "bets_per_draw": len(amounts),
        "nonzero_bets_per_draw": nz_count,
        "tiny_bet_ratio_in_draw": sum(1 for a in nonzero if a <= 1) / max(nz_count, 1),
        "max_bet_share_in_draw": max_amt / total if total > 0 else 0.0,
        "bet_amount_std_in_draw": float(np.std(nonzero)) if nz_count > 1 else 0.0,
        "bet_amount_mean_in_draw": float(np.mean(nonzero)) if nz_count > 0 else 0.0,
        "entropy_in_draw": safe_entropy(amounts),
        "gini_in_draw": max(gini_coeff(amounts), 0.0),
        "unique_positions_in_draw": nz_count,
        "position_coverage": nz_count / 38.0,
    }


def normalize_bet_position(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"", "NAN", "NONE"}:
        return None
    if raw in {"00", "000", "DOUBLE_ZERO", "DOUBLEZERO"}:
        return "00"
    if raw.isdigit():
        number = int(raw)
        if number == 0:
            return "0"
        if 1 <= number <= 36:
            return str(number)
    return raw


def bet_position_set(bets_list: Any) -> set[str]:
    positions: set[str] = set()
    if not isinstance(bets_list, list):
        return positions
    for bet in bets_list:
        try:
            amount = float(bet.get("bet_amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        if amount <= 0:
            continue
        position = normalize_bet_position(bet.get("number"))
        if position is not None:
            positions.add(position)
    return positions


def mean_pairwise_jaccard(position_sets: list[set[str]]) -> float:
    if len(position_sets) < 2:
        return 0.0
    values: list[float] = []
    for left_index in range(len(position_sets)):
        for right_index in range(left_index + 1, len(position_sets)):
            left = position_sets[left_index]
            right = position_sets[right_index]
            union = left | right
            values.append((len(left & right) / len(union)) if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def _coerce_datetime(value: Any):
    if isinstance(value, dict) and "$date" in value:
        return value["$date"]
    return value


def _normalize_timestamp(df: pd.DataFrame) -> pd.Series:
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in TIMESTAMP_CANDIDATES:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col].map(_coerce_datetime), utc=True, errors="coerce")
        ts = ts.fillna(series)
    return ts


def _mode_val(series: pd.Series):
    modes = series.mode()
    return modes.iloc[0] if len(modes) else np.nan


# ---------------------------------------------------------------------------
# Module-level feature helpers (shared by training bulk path and serving path)
# ---------------------------------------------------------------------------

def _normalize_raw_df(
    raw_df: pd.DataFrame,
    compute_inter_draw_seconds: bool = True,
    sort_rows: bool = True,
) -> pd.DataFrame:
    """Normalize a raw draw DataFrame — identical logic to FeatureEngineering._normalize()."""
    df = raw_df.copy()
    df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
    df["draw_id"] = pd.to_numeric(df["draw_id"], errors="coerce").astype("Int64")
    df["ts"] = _normalize_timestamp(df)
    df["bets_parsed"] = df["bets"].apply(parse_bets)

    draw_feats = pd.DataFrame(df["bets_parsed"].apply(compute_draw_features).tolist())
    df = pd.concat([df.reset_index(drop=True), draw_feats.reset_index(drop=True)], axis=1)

    if "win_points" in df.columns:
        df["win_points"] = pd.to_numeric(df["win_points"], errors="coerce").fillna(0.0)
    else:
        df["win_points"] = 0.0

    if "total_bet_amount" in df.columns:
        df["total_bet_amount"] = pd.to_numeric(df["total_bet_amount"], errors="coerce").fillna(0.0)
    else:
        df["total_bet_amount"] = 0.0

    if "session_id" in df.columns:
        df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce").fillna(0).astype(int)
    else:
        df["session_id"] = 0

    if "ccs_id" in df.columns:
        df["ccs_id"] = df["ccs_id"].astype(str)
    else:
        df["ccs_id"] = ""

    df["net_result"] = df["win_points"] - df["total_bet_amount"]
    df["bet_template"] = df["bets_parsed"].apply(make_bet_template_key)
    df = df.drop(columns=["bets_parsed"])

    if sort_rows:
        df = df.sort_values(["member_id", "ts", "draw_id"])
    if compute_inter_draw_seconds:
        if not sort_rows:
            df = df.sort_values(["member_id", "ts", "draw_id"])
        df["inter_draw_seconds"] = df.groupby("member_id")["ts"].diff().dt.total_seconds()

    df["fraud_event_key"] = df["draw_id"].astype(str) + "|" + df["member_id"]
    return df


def _aggregate_player_features_from_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level features from normalized draw history."""
    if history_df.empty:
        return pd.DataFrame(columns=["member_id"])

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
        primary_ccs_id=("ccs_id", _mode_val),
    ).reset_index()

    for col in ["stake_std", "entropy_std", "gini_std", "std_net_result", "std_inter_draw_seconds"]:
        player_agg[col] = player_agg[col].fillna(0)
    for col in ["avg_inter_draw_seconds", "median_inter_draw_seconds", "min_inter_draw_seconds"]:
        player_agg[col] = player_agg[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    player_agg["active_days"] = pd.to_numeric(player_agg["active_days"], errors="coerce").fillna(0).astype(int)

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
    avg_session_draws = (
        session_draws.groupby("member_id")["draws_in_session"]
        .mean()
        .reset_index(name="avg_draws_per_session")
    )
    player_agg = player_agg.merge(avg_session_draws, on="member_id", how="left")
    player_agg["avg_draws_per_session"] = player_agg["avg_draws_per_session"].fillna(1)

    max_reuse = (
        history_df.groupby("member_id")["bet_template"]
        .apply(lambda vals: vals.value_counts().iloc[0] if len(vals) else 1)
        .reset_index(name="max_template_reuse")
    )
    player_agg = player_agg.merge(max_reuse, on="member_id", how="left")

    ccs_player_count = (
        history_df.groupby("ccs_id")["member_id"].nunique().reset_index(name="ccs_player_count")
    )
    ccs_totals = (
        history_df.groupby("ccs_id")
        .agg(ccs_total_staked=("total_bet_amount", "sum"), ccs_avg_bet=("total_bet_amount", "mean"))
        .reset_index()
    )
    player_agg = (
        player_agg
        .merge(ccs_player_count.rename(columns={"ccs_id": "primary_ccs_id"}), on="primary_ccs_id", how="left")
        .merge(ccs_totals.rename(columns={"ccs_id": "primary_ccs_id"}), on="primary_ccs_id", how="left")
    )
    for col in ["ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]:
        player_agg[col] = player_agg[col].fillna(0)

    return player_agg


def _compute_draw_collusion_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row per draw with label-free cohort coverage/overlap metrics."""
    columns = [
        "draw_id",
        "cohort_size",
        "union_position_coverage",
        "mean_pairwise_jaccard",
        "total_cohort_stake",
    ]
    if history_df.empty or "draw_id" not in history_df.columns or "bets" not in history_df.columns:
        return pd.DataFrame(columns=columns)

    working = history_df[["draw_id", "member_id", "bets", "total_bet_amount"]].copy()
    working["draw_id"] = pd.to_numeric(working["draw_id"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["draw_id"])
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["member_id"] = working["member_id"].astype(str).str.strip().str.upper()
    working["_position_set"] = working["bets"].apply(lambda raw: bet_position_set(parse_bets(raw)))
    working["total_bet_amount"] = pd.to_numeric(working["total_bet_amount"], errors="coerce").fillna(0.0)

    records = []
    for draw_id, draw_rows in working.groupby("draw_id", sort=False):
        position_sets = list(draw_rows["_position_set"])
        union_positions = set().union(*position_sets) if position_sets else set()
        cohort_size = int(draw_rows["member_id"].nunique())
        records.append(
            {
                "draw_id": int(draw_id),
                "cohort_size": cohort_size,
                "union_position_coverage": float(len(union_positions) / ROULETTE_POSITION_COUNT),
                "mean_pairwise_jaccard": mean_pairwise_jaccard(position_sets),
                "total_cohort_stake": float(draw_rows["total_bet_amount"].sum()),
            }
        )
    return pd.DataFrame(records)


def _collusion_member_draw_rows(history_df: pd.DataFrame, draw_features: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty or draw_features.empty:
        return pd.DataFrame()

    keys = history_df[["member_id", "draw_id"]].copy()
    keys["member_id"] = keys["member_id"].astype(str).str.strip().str.upper()
    keys["draw_id"] = pd.to_numeric(keys["draw_id"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["draw_id"]).drop_duplicates()
    draw_metrics = draw_features.copy()
    draw_metrics["draw_id"] = pd.to_numeric(draw_metrics["draw_id"], errors="coerce").astype("Int64")
    merged = keys.merge(draw_metrics, on="draw_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=["member_id", *COLLUSION_FEATURE_COLUMNS])
    merged["cohort_2plus"] = (merged["cohort_size"].fillna(0) >= 2).astype(float)
    in_cohort = merged["cohort_size"].fillna(0) >= 2
    merged["pairwise_jaccard_for_agg"] = np.where(
        in_cohort,
        merged["mean_pairwise_jaccard"].fillna(0.0),
        np.nan,
    )
    return merged


def _aggregate_collusion_member_draw_rows(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["member_id", *COLLUSION_FEATURE_COLUMNS])

    features = merged.groupby("member_id").agg(
        max_cohort_coverage_in_draws=("union_position_coverage", "max"),
        mean_cohort_coverage_in_draws=("union_position_coverage", "mean"),
        pct_draws_in_cohort_2plus=("cohort_2plus", "mean"),
        mean_cohort_size=("cohort_size", "mean"),
        mean_pairwise_jaccard_when_in_cohort=("pairwise_jaccard_for_agg", "mean"),
    ).reset_index()

    for col in COLLUSION_FEATURE_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    return features


def _aggregate_collusion_features(history_df: pd.DataFrame, draw_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate draw-level collusion metrics back to player-level features."""
    return _aggregate_collusion_member_draw_rows(_collusion_member_draw_rows(history_df, draw_features))


def build_ccs_stats_lookup(history_df: pd.DataFrame) -> pd.DataFrame:
    """Build frozen CCS cohort stats from training history, indexed by ccs_id.

    Returns a DataFrame indexed by ccs_id with columns:
    ccs_player_count, ccs_total_staked, ccs_avg_bet.
    """
    ccs_counts = (
        history_df.groupby("ccs_id")["member_id"].nunique().reset_index(name="ccs_player_count")
    )
    ccs_sums = (
        history_df.groupby("ccs_id")
        .agg(ccs_total_staked=("total_bet_amount", "sum"), ccs_avg_bet=("total_bet_amount", "mean"))
        .reset_index()
    )
    lookup = ccs_counts.merge(ccs_sums, on="ccs_id", how="outer")
    lookup[["ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]] = (
        lookup[["ccs_player_count", "ccs_total_staked", "ccs_avg_bet"]].fillna(0)
    )
    return lookup.set_index("ccs_id")


def compute_single_player_features(
    raw_df: pd.DataFrame,
    ccs_stats_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for one player's raw draws in a training-compatible way.

    Member-local features are computed directly from raw_df.
    CCS cohort features (ccs_player_count, ccs_total_staked, ccs_avg_bet) are
    taken from the frozen ccs_stats_lookup so they reflect the training cohort,
    not just this player's single-member slice.
    """
    if raw_df.empty:
        return pd.DataFrame()

    normalized = _normalize_raw_df(raw_df, compute_inter_draw_seconds=True, sort_rows=True)
    if normalized.empty:
        return pd.DataFrame()

    player_features = _aggregate_player_features_from_history(normalized)
    if player_features.empty:
        return pd.DataFrame()

    # Override the cohort-derived CCS columns with the frozen training-time lookup.
    if ccs_stats_lookup is not None and not ccs_stats_lookup.empty:
        primary_ccs = str(player_features["primary_ccs_id"].iloc[0])
        if primary_ccs in ccs_stats_lookup.index:
            row = ccs_stats_lookup.loc[primary_ccs]
            player_features["ccs_player_count"] = float(row["ccs_player_count"])
            player_features["ccs_total_staked"] = float(row["ccs_total_staked"])
            player_features["ccs_avg_bet"] = float(row["ccs_avg_bet"])
        else:
            player_features["ccs_player_count"] = 0.0
            player_features["ccs_total_staked"] = 0.0
            player_features["ccs_avg_bet"] = 0.0

    return player_features


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureEngineering:
    def __init__(
        self,
        config: FeatureEngineeringConfig,
        ingestion_artifact: DataIngestionArtifact,
        _force_mode: str = "auto",
    ):
        self.config = config
        self.ingestion_artifact = ingestion_artifact
        if _force_mode not in {"auto", "in_memory", "bucketed"}:
            raise ValueError(
                f"_force_mode must be one of 'auto', 'in_memory', 'bucketed'; got {_force_mode!r}"
            )
        self._force_mode = _force_mode

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        logger.info(
            "FeatureEngineering: starting (mode=%s, force_mode=%s)",
            self.config.mode, self._force_mode,
        )
        try:
            ensure_dir(self.config.output_dir)

            if self._force_mode == "in_memory":
                return self._initiate_feature_engineering_in_memory()
            if self._force_mode == "bucketed":
                return self._initiate_feature_engineering_bucketed()

            if self.ingestion_artifact.row_count > FEATURE_ENGINEERING_IN_MEMORY_MAX_ROWS:
                return self._initiate_feature_engineering_bucketed()

            return self._initiate_feature_engineering_in_memory()
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def _append_dataframe_to_parquet(
        df: pd.DataFrame,
        path,
        writers: dict,
    ) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer = writers.get(path)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(path), table.schema)
            writers[path] = writer
        writer.write_table(table)

    @staticmethod
    def _close_writers(writers: dict) -> None:
        for writer in writers.values():
            writer.close()
        writers.clear()

    @staticmethod
    def _choose_bucket_count(row_count: int) -> int:
        if row_count <= FEATURE_ENGINEERING_TARGET_ROWS_PER_BUCKET:
            return 1

        target_bucket_count = max(1, int(np.ceil(row_count / FEATURE_ENGINEERING_TARGET_ROWS_PER_BUCKET)))
        bucket_count = 1
        while bucket_count < target_bucket_count and bucket_count < FEATURE_ENGINEERING_MAX_BUCKETS:
            bucket_count *= 2
        return min(bucket_count, FEATURE_ENGINEERING_MAX_BUCKETS)

    @staticmethod
    def _ensure_history_schema(df: pd.DataFrame) -> pd.DataFrame:
        history_df = df.copy()
        for col in TRAINING_HISTORY_COLUMNS:
            if col in history_df.columns:
                continue
            if col == "is_fraud_player":
                history_df[col] = pd.Series(0, index=history_df.index, dtype="int64")
            elif col == "event_label":
                history_df[col] = pd.Series(0, index=history_df.index, dtype="int64")
            elif col == "first_fraud_draw_id":
                history_df[col] = pd.Series(pd.NA, index=history_df.index, dtype="Int64")
            elif col == "first_fraud_ts":
                history_df[col] = pd.Series(pd.NaT, index=history_df.index, dtype="datetime64[ns, UTC]")

        for col in HISTORY_TS_COLUMNS:
            if col in history_df.columns:
                history_df[col] = pd.to_datetime(history_df[col], errors="coerce", utc=True)

        if "first_fraud_draw_id" in history_df.columns:
            history_df["first_fraud_draw_id"] = pd.to_numeric(
                history_df["first_fraud_draw_id"],
                errors="coerce",
            ).astype("Int64")

        if "event_label" in history_df.columns:
            history_df["event_label"] = pd.to_numeric(
                history_df["event_label"],
                errors="coerce",
            ).fillna(0).astype("int64")

        if "is_fraud_player" in history_df.columns:
            history_df["is_fraud_player"] = pd.to_numeric(
                history_df["is_fraud_player"],
                errors="coerce",
            ).fillna(0).astype("int64")

        return history_df

    def _finalize_outputs(
        self,
        raw_rows: int,
        history_rows: int,
        history_path,
        player_features: pd.DataFrame,
        fraud_player_count: int,
        dropped_positive_count: int,
        fraud_funnel: dict | None = None,
        draw_features_path: Any = None,
    ) -> FeatureEngineeringArtifact:
        feature_cols = [
            c for c in player_features.columns
            if c not in set(self.config.exclude_cols) and pd.api.types.is_numeric_dtype(player_features[c])
        ]

        player_path = self.config.output_dir / "player_features.parquet"
        summary_path = self.config.output_dir / "feature_summary.json"

        save_parquet(player_features, player_path)

        summary = {
            "mode": self.config.mode,
            "raw_rows": raw_rows,
            "history_rows": history_rows,
            "player_count": len(player_features),
            "fraud_player_count": fraud_player_count,
            "dropped_positive_count": dropped_positive_count,
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
            "compute_collusion_features": bool(self.config.compute_collusion_features),
            "draw_features_path": str(draw_features_path) if draw_features_path else None,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        if fraud_funnel is not None:
            summary["fraud_funnel"] = fraud_funnel
            funnel_path = self.config.output_dir / "fraud_funnel.json"
            write_json(fraud_funnel, funnel_path)
            logger.info(
                "FraudFunnel: csv_rows=%s csv_unique_members=%s use_window=%s "
                "window_days=%s members_in_raw=%s members_in_window=%s "
                "fraud_member_history_rows=%s players_in_feature_table=%s "
                "dropped_positive=%s",
                fraud_funnel.get("csv_rows"),
                fraud_funnel.get("csv_unique_members"),
                fraud_funnel.get("use_window"),
                fraud_funnel.get("window_days"),
                fraud_funnel.get("members_in_raw"),
                fraud_funnel.get("members_in_window"),
                fraud_funnel.get("fraud_member_history_rows"),
                fraud_funnel.get("players_in_feature_table"),
                fraud_funnel.get("dropped_positive_count"),
            )
        write_json(summary, summary_path)

        logger.info("FeatureEngineering: complete — saved to %s", self.config.output_dir)
        return FeatureEngineeringArtifact(
            player_features_path=player_path,
            history_df_path=history_path,
            fraud_player_count=fraud_player_count,
            dropped_positive_count=dropped_positive_count,
            feature_columns=feature_cols,
            feature_summary_path=summary_path,
            mode=self.config.mode,
            draw_features_path=draw_features_path,
        )

    def _initiate_feature_engineering_in_memory(self) -> FeatureEngineeringArtifact:
        raw_df = pd.read_parquet(self.ingestion_artifact.raw_data_path)
        logger.info("Loaded raw data: %d rows", len(raw_df))

        normalized_df = self._normalize(raw_df)

        fraud_player_count = 0
        dropped_positive_count = 0
        funnel: dict | None = None

        if self.config.mode == "training_eval":
            fraud_df = self._load_fraud_csv()
            history_df, _, _ = self._apply_training_eval_steps(normalized_df, fraud_df)
        else:
            fraud_df = None
            history_df = normalized_df.copy()

        player_features = self._aggregate_player_features(history_df)
        draw_features_path = None
        if self.config.compute_collusion_features:
            draw_features = _compute_draw_collusion_features(history_df)
            draw_features_path = self.config.output_dir / "draw_features.parquet"
            save_parquet(draw_features, draw_features_path)
            collusion_features = _aggregate_collusion_features(history_df, draw_features)
            player_features = player_features.merge(collusion_features, on="member_id", how="left")
            for col in COLLUSION_FEATURE_COLUMNS:
                player_features[col] = pd.to_numeric(player_features[col], errors="coerce").fillna(0.0)

        if self.config.mode == "training_eval" and fraud_df is not None:
            has_dates = bool(fraud_df["fraud_date"].notna().any())
            window_days = int(self.config.fraud_label_window_days)
            use_window = has_dates and window_days > 0

            if use_window:
                fraud_players = set(fraud_df["member_id_norm"])
            else:
                fraud_event_keys = set(fraud_df["fraud_event_key"])
                fraud_players = set(
                    normalized_df.loc[normalized_df["fraud_event_key"].isin(fraud_event_keys), "member_id"]
                )

            player_features["event_fraud_flag"] = player_features["member_id"].isin(fraud_players).astype(int)
            fraud_player_count = int(player_features["event_fraud_flag"].sum())
            dropped_positive_count = len(fraud_players) - fraud_player_count

            funnel = self._compute_fraud_funnel(
                fraud_df=fraud_df,
                normalized_members=set(normalized_df["member_id"].unique()),
                history_fraud_member_rows=int(history_df.get("is_fraud_player", pd.Series(dtype=int)).eq(1).sum()),
                player_features=player_features,
                use_window=use_window,
                window_days=window_days,
            )

        logger.info(
            "FeatureEngineering: %d players, %d fraud, %d dropped positive",
            len(player_features), fraud_player_count, dropped_positive_count,
        )

        history_path = self.config.output_dir / "history_df.parquet"
        save_parquet(self._ensure_history_schema(history_df), history_path)

        return self._finalize_outputs(
            raw_rows=len(raw_df),
            history_rows=len(history_df),
            history_path=history_path,
            player_features=player_features,
            fraud_player_count=fraud_player_count,
            dropped_positive_count=dropped_positive_count,
            fraud_funnel=funnel,
            draw_features_path=draw_features_path,
        )

    def _initiate_feature_engineering_bucketed(self) -> FeatureEngineeringArtifact:
        logger.info(
            "FeatureEngineering: using bucketed parquet workflow for %d raw rows",
            self.ingestion_artifact.row_count,
        )

        history_path = self.config.output_dir / "history_df.parquet"
        draw_features_path = self.config.output_dir / "draw_features.parquet" if self.config.compute_collusion_features else None
        raw_path = self.ingestion_artifact.raw_data_path
        bucket_dir = ensure_dir(self.config.output_dir / "_bucketed_normalized")
        draw_bucket_dir = ensure_dir(self.config.output_dir / "_bucketed_draws") if self.config.compute_collusion_features else None
        bucket_count = self._choose_bucket_count(self.ingestion_artifact.row_count)
        bucket_paths = [bucket_dir / f"bucket_{bucket_id:03d}.parquet" for bucket_id in range(bucket_count)]
        draw_bucket_paths = (
            [draw_bucket_dir / f"draw_bucket_{bucket_id:03d}.parquet" for bucket_id in range(bucket_count)]
            if draw_bucket_dir is not None else []
        )

        parquet_file = pq.ParquetFile(raw_path)
        requested_columns = [
            column for column in FEATURE_ENGINEERING_RAW_COLUMNS if column in parquet_file.schema_arrow.names
        ]

        raw_rows = 0
        bucket_writers: dict = {}
        try:
            for batch_index, batch in enumerate(
                parquet_file.iter_batches(columns=requested_columns, batch_size=FEATURE_ENGINEERING_BATCH_SIZE),
                start=1,
            ):
                batch_df = batch.to_pandas()
                raw_rows += len(batch_df)
                normalized_batch = self._normalize(
                    batch_df,
                    compute_inter_draw_seconds=False,
                    sort_rows=False,
                )
                if normalized_batch.empty:
                    continue

                bucket_ids = (
                    pd.util.hash_pandas_object(normalized_batch["member_id"], index=False)
                    .astype("uint64")
                    .to_numpy()
                    % bucket_count
                )
                normalized_batch["_bucket_id"] = bucket_ids

                for bucket_id, bucket_df in normalized_batch.groupby("_bucket_id", sort=False):
                    self._append_dataframe_to_parquet(
                        bucket_df.drop(columns="_bucket_id"),
                        bucket_paths[int(bucket_id)],
                        bucket_writers,
                    )

                if batch_index % 10 == 0:
                    logger.info(
                        "FeatureEngineering: partitioned %d raw rows into %d bucket(s)",
                        raw_rows,
                        bucket_count,
                    )
        finally:
            self._close_writers(bucket_writers)

        fraud_df = self._load_fraud_csv() if self.config.mode == "training_eval" else None
        fraud_event_keys = set(fraud_df["fraud_event_key"]) if fraud_df is not None else set()
        # When DATE is present, "fraud member" = CSV membership directly. The
        # whole CSV member set is known up front, so we don't need to scan
        # buckets to discover it. Keeping ``fraud_players_seen`` for the
        # legacy code path (DATE missing) where we still need to scan bucket
        # rows for matches.
        if fraud_df is not None and bool(fraud_df["fraud_date"].notna().any()) and int(self.config.fraud_label_window_days) > 0:
            use_window_bucketed = True
            fraud_players_csv: set[str] = set(fraud_df["member_id_norm"])
        else:
            use_window_bucketed = False
            fraud_players_csv = set()
        fraud_players_seen: set[str] = set()
        # For the funnel: members from CSV that appeared in any normalized
        # bucket (any row, before window filtering).
        fraud_members_in_raw: set[str] = set()
        # For the funnel: members from CSV that had at least one row
        # surviving the window filter (history_df, is_fraud_player=1).
        fraud_members_in_window: set[str] = set()
        fraud_member_history_rows = 0
        history_rows = 0
        player_feature_frames: list[pd.DataFrame] = []
        history_writers: dict = {}
        draw_bucket_writers: dict = {}

        try:
            for bucket_index, bucket_path in enumerate(bucket_paths, start=1):
                if not bucket_path.exists():
                    continue

                bucket_df = pd.read_parquet(bucket_path)
                if bucket_df.empty:
                    bucket_path.unlink(missing_ok=True)
                    continue

                bucket_df = self._finalize_normalized_bucket(bucket_df)

                if self.config.mode == "training_eval":
                    if use_window_bucketed:
                        fraud_members_in_raw.update(
                            set(bucket_df["member_id"].unique()).intersection(fraud_players_csv)
                        )
                    else:
                        fraud_players_seen.update(
                            bucket_df.loc[bucket_df["fraud_event_key"].isin(fraud_event_keys), "member_id"]
                        )
                    history_df, _, _ = self._apply_training_eval_steps(bucket_df, fraud_df)
                    if "is_fraud_player" in history_df.columns:
                        bucket_fraud_rows = int(history_df["is_fraud_player"].eq(1).sum())
                        fraud_member_history_rows += bucket_fraud_rows
                        if bucket_fraud_rows:
                            fraud_members_in_window.update(
                                history_df.loc[history_df["is_fraud_player"].eq(1), "member_id"].unique()
                            )
                    history_df = self._ensure_history_schema(history_df)
                else:
                    history_df = bucket_df.copy()

                player_features_bucket = self._aggregate_player_features(history_df)
                if not player_features_bucket.empty:
                    player_feature_frames.append(player_features_bucket)

                history_rows += len(history_df)
                if self.config.compute_collusion_features and not history_df.empty:
                    draw_cols = [col for col in ["member_id", "draw_id", "bets", "total_bet_amount"] if col in history_df.columns]
                    draw_input = history_df[draw_cols].copy()
                    draw_ids = pd.to_numeric(draw_input["draw_id"], errors="coerce")
                    valid_draw_id_mask = draw_ids.notna()
                    draw_input = draw_input.loc[valid_draw_id_mask].copy()
                    if not draw_input.empty:
                        draw_ids = draw_ids.loc[valid_draw_id_mask].astype("int64")
                        draw_bucket_ids = (draw_ids.to_numpy() % bucket_count).astype("int64")
                        draw_input["_draw_bucket_id"] = draw_bucket_ids
                        for draw_bucket_id, draw_bucket_df in draw_input.groupby("_draw_bucket_id", sort=False):
                            self._append_dataframe_to_parquet(
                                draw_bucket_df.drop(columns="_draw_bucket_id"),
                                draw_bucket_paths[int(draw_bucket_id)],
                                draw_bucket_writers,
                            )
                self._append_dataframe_to_parquet(history_df, history_path, history_writers)
                bucket_path.unlink(missing_ok=True)

                if bucket_index % 10 == 0 or bucket_index == bucket_count:
                    logger.info(
                        "FeatureEngineering: processed %d/%d bucket(s), history rows=%d, players so far=%d",
                        bucket_index,
                        bucket_count,
                        history_rows,
                        sum(len(frame) for frame in player_feature_frames),
                    )
        finally:
            self._close_writers(history_writers)
            self._close_writers(draw_bucket_writers)
            shutil.rmtree(bucket_dir, ignore_errors=True)

        if player_feature_frames:
            player_features = pd.concat(player_feature_frames, ignore_index=True)
        else:
            player_features = pd.DataFrame(columns=["member_id"])

        if self.config.compute_collusion_features and draw_features_path is not None:
            draw_feature_frames: list[pd.DataFrame] = []
            collusion_member_draw_frames: list[pd.DataFrame] = []
            for draw_bucket_path in draw_bucket_paths:
                if not draw_bucket_path.exists():
                    continue
                draw_history_df = pd.read_parquet(draw_bucket_path)
                draw_features_bucket = _compute_draw_collusion_features(draw_history_df)
                if not draw_features_bucket.empty:
                    draw_feature_frames.append(draw_features_bucket)
                    member_draw_rows = _collusion_member_draw_rows(draw_history_df, draw_features_bucket)
                    if not member_draw_rows.empty:
                        collusion_member_draw_frames.append(member_draw_rows)
                draw_bucket_path.unlink(missing_ok=True)
            shutil.rmtree(draw_bucket_dir, ignore_errors=True)

            draw_features = (
                pd.concat(draw_feature_frames, ignore_index=True)
                if draw_feature_frames else pd.DataFrame(columns=[
                    "draw_id", "cohort_size", "union_position_coverage",
                    "mean_pairwise_jaccard", "total_cohort_stake",
                ])
            )
            save_parquet(draw_features, draw_features_path)
            collusion_features = (
                _aggregate_collusion_member_draw_rows(pd.concat(collusion_member_draw_frames, ignore_index=True))
                if collusion_member_draw_frames else pd.DataFrame(columns=["member_id", *COLLUSION_FEATURE_COLUMNS])
            )
            player_features = player_features.merge(collusion_features, on="member_id", how="left")
            for col in COLLUSION_FEATURE_COLUMNS:
                player_features[col] = pd.to_numeric(player_features[col], errors="coerce").fillna(0.0)
        elif draw_bucket_dir is not None:
            shutil.rmtree(draw_bucket_dir, ignore_errors=True)

        fraud_player_count = 0
        dropped_positive_count = 0
        funnel: dict | None = None
        if self.config.mode == "training_eval" and fraud_df is not None:
            fraud_players = fraud_players_csv if use_window_bucketed else fraud_players_seen
            player_features["event_fraud_flag"] = player_features["member_id"].isin(fraud_players).astype(int)
            fraud_player_count = int(player_features["event_fraud_flag"].sum())
            dropped_positive_count = len(fraud_players) - fraud_player_count

            funnel = {
                "csv_rows": int(len(fraud_df)),
                "csv_unique_members": int(fraud_df["member_id_norm"].nunique()),
                "csv_with_valid_date": int(fraud_df["fraud_date"].notna().sum()),
                "use_window": use_window_bucketed,
                "window_days": int(self.config.fraud_label_window_days),
                "members_in_raw": (
                    len(fraud_members_in_raw) if use_window_bucketed
                    else len(fraud_players_seen)
                ),
                "members_in_window": (
                    len(fraud_members_in_window) if use_window_bucketed
                    else len(fraud_players_seen)
                ),
                "fraud_member_history_rows": fraud_member_history_rows,
                "players_in_feature_table": int(fraud_player_count),
                "dropped_positive_count": int(dropped_positive_count),
            }

        logger.info(
            "FeatureEngineering: %d players, %d fraud, %d dropped positive",
            len(player_features), fraud_player_count, dropped_positive_count,
        )

        if not history_path.exists():
            save_parquet(pd.DataFrame(columns=["member_id"]), history_path)

        return self._finalize_outputs(
            raw_rows=raw_rows,
            history_rows=history_rows,
            history_path=history_path,
            player_features=player_features,
            fraud_player_count=fraud_player_count,
            dropped_positive_count=dropped_positive_count,
            fraud_funnel=funnel,
            draw_features_path=draw_features_path,
        )

    def _load_fraud_csv(self) -> pd.DataFrame:
        """Load and normalize the fraud CSV.

        When a `DATE` column is present, parses it into a UTC `fraud_date`
        timestamp used for weekly window matching in
        ``_apply_training_eval_steps``. When `DATE` is absent, ``fraud_date``
        is set to NaT and the matching path falls back to the legacy
        whole-window set-membership behavior.
        """
        fraud_df = pd.read_csv(self.config.fraud_csv_path)
        fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]
        fraud_df["member_id_norm"] = fraud_df["member_id"].astype(str).str.strip().str.upper()
        fraud_df["draw_id_norm"] = pd.to_numeric(fraud_df["draw_id"], errors="coerce").astype("Int64")
        fraud_df["fraud_event_key"] = (
            fraud_df["draw_id_norm"].astype(str) + "|" + fraud_df["member_id_norm"]
        )
        if "date" in fraud_df.columns:
            fraud_df["fraud_date"] = pd.to_datetime(fraud_df["date"], errors="coerce", utc=True)
            unparseable = int(fraud_df["fraud_date"].isna().sum())
            if unparseable:
                logger.warning(
                    "FraudCSV: %d/%d rows have unparseable DATE — dropped from labeling",
                    unparseable, len(fraud_df),
                )
                fraud_df = fraud_df.loc[fraud_df["fraud_date"].notna()].copy()
        else:
            logger.warning(
                "FraudCSV: no DATE column — falling back to whole-window labeling. "
                "Add a DATE column to enable weekly matching."
            )
            fraud_df["fraud_date"] = pd.NaT
        return fraud_df

    def _compute_fraud_funnel(
        self,
        fraud_df: pd.DataFrame,
        normalized_members: set,
        history_fraud_member_rows: int,
        player_features: pd.DataFrame,
        use_window: bool,
        window_days: int,
    ) -> dict:
        """Build the fraud-label funnel report for the in-memory training path.

        Captures the count drop from CSV rows to labeled players in the final
        feature table — the answer to "where did my fraud labels go." The
        bucketed orchestration computes its own funnel in-line because it
        needs to aggregate counts across buckets without ever materializing
        the full normalized frame.
        """
        csv_members = set(fraud_df["member_id_norm"])
        members_in_raw = len(csv_members.intersection(normalized_members))
        if "event_fraud_flag" in player_features.columns:
            in_table = int(player_features["event_fraud_flag"].sum())
        else:
            in_table = 0
        return {
            "csv_rows": int(len(fraud_df)),
            "csv_unique_members": int(fraud_df["member_id_norm"].nunique()),
            "csv_with_valid_date": int(fraud_df["fraud_date"].notna().sum()),
            "use_window": bool(use_window),
            "window_days": int(window_days),
            "members_in_raw": int(members_in_raw),
            # In-memory mode: we count fraud-member rows surviving the cutoff
            # as the "in window" signal. A member with at least one surviving
            # row will appear in player_features (counted in members_in_raw
            # already); the row count is the load-bearing diagnostic.
            "members_in_window": int(in_table),
            "fraud_member_history_rows": int(history_fraud_member_rows),
            "players_in_feature_table": int(in_table),
            "dropped_positive_count": int(len(csv_members) - in_table),
        }

    def _normalize(
        self,
        raw_df: pd.DataFrame,
        compute_inter_draw_seconds: bool = True,
        sort_rows: bool = True,
    ) -> pd.DataFrame:
        return _normalize_raw_df(raw_df, compute_inter_draw_seconds=compute_inter_draw_seconds, sort_rows=sort_rows)

    def _finalize_normalized_bucket(self, bucket_df: pd.DataFrame) -> pd.DataFrame:
        df = bucket_df.copy()
        df = df.sort_values(["member_id", "ts", "draw_id"])
        df["inter_draw_seconds"] = df.groupby("member_id")["ts"].diff().dt.total_seconds()
        return df

    def _apply_training_eval_steps(
        self, normalized_df: pd.DataFrame, fraud_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int, int]:
        """Tag fraud rows and apply pre-fraud cutoff.

        Two label-matching paths:

        - **Weekly (DATE present, fraud_label_window_days > 0):** a row's
          ``event_label`` is 1 iff its ``(draw_id, member_id)`` is in the CSV
          AND its ``ts`` falls in ``[fraud_date - window, fraud_date)``. The
          fraud member's history is then narrowed to the same window
          ``[member_first_fraud_date - window, member_first_fraud_date)``.
          Outside that window we treat the member_id as a different person
          (fraud-ops recycles flagged ids weekly).

        - **Legacy (DATE missing or window=0):** set membership on
          ``fraud_event_key``; pre-fraud cutoff uses the earliest fraud row
          observed in raw data as ``first_fraud_ts``.
        """
        df = normalized_df.copy()

        has_dates = bool(fraud_df["fraud_date"].notna().any())
        window_days = int(self.config.fraud_label_window_days)
        use_window = has_dates and window_days > 0

        if use_window:
            window = pd.Timedelta(days=window_days)
            fraud_event_lookup = (
                fraud_df.dropna(subset=["fraud_date"])
                .groupby("fraud_event_key", as_index=False)["fraud_date"]
                .min()
                .rename(columns={"fraud_date": "_event_fraud_date"})
            )
            df = df.merge(fraud_event_lookup, on="fraud_event_key", how="left")
            in_window = (
                df["_event_fraud_date"].notna()
                & df["ts"].notna()
                & (df["ts"] >= df["_event_fraud_date"] - window)
                & (df["ts"] < df["_event_fraud_date"])
            )
            df["event_label"] = in_window.astype(int)
            # A member is "fraud" if they appear in the CSV at all — the
            # window only restricts which DRAWS count as the fraud event and
            # which ROWS count as their pre-fraud history. CSV membership is
            # ground truth for "this id was caught."
            fraud_players = set(fraud_df["member_id_norm"])
        else:
            fraud_event_keys = set(fraud_df["fraud_event_key"])
            df["event_label"] = df["fraud_event_key"].isin(fraud_event_keys).astype(int)
            fraud_players = set(df.loc[df["event_label"] == 1, "member_id"])

        df["is_fraud_player"] = df["member_id"].isin(fraud_players).astype(int)

        if not fraud_players or not self.config.apply_pre_fraud_cutoff:
            history_df = df.copy()
            if "_event_fraud_date" in history_df.columns:
                history_df = history_df.drop(columns=["_event_fraud_date"])
            return history_df, len(fraud_players), 0

        if use_window:
            # Per-member earliest fraud_date from CSV, then narrow each fraud
            # member's history to ``[first_fraud_ts - window, first_fraud_ts)``.
            member_first = (
                fraud_df.dropna(subset=["fraud_date"])
                .groupby("member_id_norm", as_index=False)["fraud_date"]
                .min()
                .rename(columns={
                    "member_id_norm": "member_id",
                    "fraud_date": "first_fraud_ts",
                })
            )
            df = df.merge(member_first, on="member_id", how="left")
            # ``first_fraud_draw_id`` is preserved for downstream schema
            # compatibility but not derived from CSV — set NA. The legacy
            # path (below) populates it.
            df["first_fraud_draw_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

            is_fraud_player_mask = df["is_fraud_player"].eq(1)
            in_catch_window = (
                df["ts"].notna()
                & df["first_fraud_ts"].notna()
                & (df["ts"] >= df["first_fraud_ts"] - window)
                & (df["ts"] < df["first_fraud_ts"])
            )
            non_fraud_mask = ~is_fraud_player_mask
            history_df = df.loc[(is_fraud_player_mask & in_catch_window) | non_fraud_mask].copy()
        else:
            # Compute first fraud event per player.
            # The earliest row is selected by (ts, draw_id) sort with NaT last, so
            # first_fraud_ts and first_fraud_draw_id both come from the SAME event row.
            # Using independent min() per column would mix fields from different rows
            # whenever a member has multiple fraud events and the earliest timestamp
            # is not on the same row as the lowest draw_id.
            event_match_df = (
                df.loc[df["event_label"] == 1, ["member_id", "draw_id", "ts"]]
                .sort_values(["member_id", "ts", "draw_id"], na_position="last")
            )
            first_fraud = (
                event_match_df
                .groupby("member_id", as_index=False)
                .agg(
                    first_fraud_ts=("ts", "first"),
                    first_fraud_draw_id=("draw_id", "first"),
                )
            )
            df = df.merge(
                first_fraud[["member_id", "first_fraud_ts", "first_fraud_draw_id"]],
                on="member_id",
                how="left",
            )

            pre_fraud_mask = df["is_fraud_player"].eq(1) & (
                (df["ts"].notna() & df["first_fraud_ts"].notna() & (df["ts"] < df["first_fraud_ts"]))
                | (df["first_fraud_ts"].isna() & df["first_fraud_draw_id"].notna() & (df["draw_id"] < df["first_fraud_draw_id"]))
            )
            non_fraud_mask = df["is_fraud_player"].eq(0)
            history_df = df.loc[pre_fraud_mask | non_fraud_mask].copy()

        if "_event_fraud_date" in history_df.columns:
            history_df = history_df.drop(columns=["_event_fraud_date"])

        return history_df, len(fraud_players), 0  # dropped count computed after aggregation

    def _aggregate_player_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        return _aggregate_player_features_from_history(history_df)
