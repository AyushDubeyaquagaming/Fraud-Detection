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
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
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
        )

    def _initiate_feature_engineering_in_memory(self) -> FeatureEngineeringArtifact:
        raw_df = pd.read_parquet(self.ingestion_artifact.raw_data_path)
        logger.info("Loaded raw data: %d rows", len(raw_df))

        normalized_df = self._normalize(raw_df)

        fraud_player_count = 0
        dropped_positive_count = 0

        if self.config.mode == "training_eval":
            fraud_df = self._load_fraud_csv()
            history_df, fraud_player_count, dropped_positive_count = self._apply_training_eval_steps(
                normalized_df, fraud_df
            )
        else:
            history_df = normalized_df.copy()

        player_features = self._aggregate_player_features(history_df)

        if self.config.mode == "training_eval":
            fraud_df = self._load_fraud_csv()
            fraud_event_keys = set(fraud_df["fraud_event_key"])
            fraud_players = set(
                normalized_df.loc[normalized_df["fraud_event_key"].isin(fraud_event_keys), "member_id"]
            )
            player_features["event_fraud_flag"] = player_features["member_id"].isin(fraud_players).astype(int)
            fraud_player_count = int(player_features["event_fraud_flag"].sum())
            dropped_positive_count = len(fraud_players) - fraud_player_count

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
        )

    def _initiate_feature_engineering_bucketed(self) -> FeatureEngineeringArtifact:
        logger.info(
            "FeatureEngineering: using bucketed parquet workflow for %d raw rows",
            self.ingestion_artifact.row_count,
        )

        history_path = self.config.output_dir / "history_df.parquet"
        raw_path = self.ingestion_artifact.raw_data_path
        bucket_dir = ensure_dir(self.config.output_dir / "_bucketed_normalized")
        bucket_count = self._choose_bucket_count(self.ingestion_artifact.row_count)
        bucket_paths = [bucket_dir / f"bucket_{bucket_id:03d}.parquet" for bucket_id in range(bucket_count)]

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
        fraud_players_seen: set[str] = set()
        history_rows = 0
        player_feature_frames: list[pd.DataFrame] = []
        history_writers: dict = {}

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
                    fraud_players_seen.update(
                        bucket_df.loc[bucket_df["fraud_event_key"].isin(fraud_event_keys), "member_id"]
                    )
                    history_df, _, _ = self._apply_training_eval_steps(bucket_df, fraud_df)
                    history_df = self._ensure_history_schema(history_df)
                else:
                    history_df = bucket_df.copy()

                player_features_bucket = self._aggregate_player_features(history_df)
                if not player_features_bucket.empty:
                    player_feature_frames.append(player_features_bucket)

                history_rows += len(history_df)
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
            shutil.rmtree(bucket_dir, ignore_errors=True)

        if player_feature_frames:
            player_features = pd.concat(player_feature_frames, ignore_index=True)
        else:
            player_features = pd.DataFrame(columns=["member_id"])

        fraud_player_count = 0
        dropped_positive_count = 0
        if self.config.mode == "training_eval":
            player_features["event_fraud_flag"] = player_features["member_id"].isin(fraud_players_seen).astype(int)
            fraud_player_count = int(player_features["event_fraud_flag"].sum())
            dropped_positive_count = len(fraud_players_seen) - fraud_player_count

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
        )

    def _load_fraud_csv(self) -> pd.DataFrame:
        fraud_df = pd.read_csv(self.config.fraud_csv_path)
        fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]
        fraud_df["member_id_norm"] = fraud_df["member_id"].astype(str).str.strip().str.upper()
        fraud_df["draw_id_norm"] = pd.to_numeric(fraud_df["draw_id"], errors="coerce").astype("Int64")
        fraud_df["fraud_event_key"] = (
            fraud_df["draw_id_norm"].astype(str) + "|" + fraud_df["member_id_norm"]
        )
        return fraud_df

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
        fraud_event_keys = set(fraud_df["fraud_event_key"])
        df = normalized_df.copy()
        df["event_label"] = df["fraud_event_key"].isin(fraud_event_keys).astype(int)

        fraud_players = set(df.loc[df["event_label"] == 1, "member_id"])

        if not fraud_players or not self.config.apply_pre_fraud_cutoff:
            history_df = df.copy()
            return history_df, len(fraud_players), 0

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

        return history_df, len(fraud_players), 0  # dropped count computed after aggregation

    def _aggregate_player_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        return _aggregate_player_features_from_history(history_df)
