from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from fraud_detection.entity.artifact_entity import DataIngestionArtifact, FeatureEngineeringArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, save_parquet, write_json

logger = get_logger(__name__)


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
    candidates = [
        "createdAt.$date", "createdat.$date", "trans_date.$date",
        "updatedAt.$date", "ts", "createdAt", "trans_date", "updatedAt",
    ]
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col].map(_coerce_datetime), utc=True, errors="coerce")
        ts = ts.fillna(series)
    return ts


def _mode_val(series: pd.Series):
    modes = series.mode()
    return modes.iloc[0] if len(modes) else np.nan


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        logger.info("FeatureEngineering: starting (mode=%s)", self.config.mode)
        try:
            ensure_dir(self.config.output_dir)

            raw_df = pd.read_parquet(self.ingestion_artifact.raw_data_path)
            logger.info("Loaded raw data: %d rows", len(raw_df))

            # Step 1: normalize
            normalized_df = self._normalize(raw_df)

            fraud_player_count = 0
            dropped_positive_count = 0

            if self.config.mode == "training_eval":
                fraud_df = self._load_fraud_csv()
                history_df, fraud_player_count, dropped_positive_count = self._apply_training_eval_steps(
                    normalized_df, fraud_df
                )
            else:
                # operational: no label dependency
                history_df = normalized_df.copy()

            # Step 2: player-level aggregation
            player_features = self._aggregate_player_features(history_df)

            if self.config.mode == "training_eval":
                # attach fraud labels
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

            # Identify feature columns (numeric, non-excluded)
            exclude = set(self.config.exclude_cols)
            feature_cols = [
                c for c in player_features.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(player_features[c])
            ]

            history_path = self.config.output_dir / "history_df.parquet"
            player_path = self.config.output_dir / "player_features.parquet"
            summary_path = self.config.output_dir / "feature_summary.json"

            # Convert bet_template (tuple) to string for parquet serialization
            history_to_save = history_df.copy()
            if "bet_template" in history_to_save.columns:
                history_to_save["bet_template"] = history_to_save["bet_template"].astype(str)
            save_parquet(history_to_save, history_path)
            save_parquet(player_features, player_path)

            summary = {
                "mode": self.config.mode,
                "raw_rows": len(raw_df),
                "history_rows": len(history_df),
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
        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def _load_fraud_csv(self) -> pd.DataFrame:
        fraud_df = pd.read_csv(self.config.fraud_csv_path)
        fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]
        fraud_df["member_id_norm"] = fraud_df["member_id"].astype(str).str.strip().str.upper()
        fraud_df["draw_id_norm"] = pd.to_numeric(fraud_df["draw_id"], errors="coerce").astype("Int64")
        fraud_df["fraud_event_key"] = (
            fraud_df["draw_id_norm"].astype(str) + "|" + fraud_df["member_id_norm"]
        )
        return fraud_df

    def _normalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
        df["draw_id"] = pd.to_numeric(df["draw_id"], errors="coerce").astype("Int64")
        df["ts"] = _normalize_timestamp(df)
        df["bets_parsed"] = df["bets"].apply(parse_bets)

        draw_feats = pd.DataFrame(df["bets_parsed"].apply(compute_draw_features).tolist())
        df = pd.concat([df.reset_index(drop=True), draw_feats.reset_index(drop=True)], axis=1)

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

        # Compute first fraud event per player
        event_match_df = df.loc[df["event_label"] == 1, ["member_id", "draw_id", "ts"]].copy()
        first_fraud = (
            event_match_df.sort_values(["member_id", "ts", "draw_id"])
            .groupby("member_id", as_index=False)
            .agg(
                first_fraud_ts=("ts", "min"),
                first_fraud_draw_id=("draw_id", "min"),
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

        # Ensure active_days is numeric (lambda may return DatetimeArray dtype on edge cases)
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
