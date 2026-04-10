from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hybrid_inference import (
    ARTIFACT_PATH,
    FEATURE_COLUMNS,
    ROOT,
    SCORED_PATH,
    aggregate_member_features,
    apply_pre_fraud_cutoff,
    load_fraud_labels,
    normalize_member_history,
)


DATA_DIR = ROOT / "data_cache"
RAW_PATH = DATA_DIR / "fraud_modeling_pull.parquet"
PLAYER_FEATURE_PATH = DATA_DIR / "player_feature_table.parquet"
HYBRID_EVAL_PATH = DATA_DIR / "hybrid_evaluation.json"


def top_capture(series: pd.Series, labels: pd.Series, pct: float) -> int:
    threshold = series.quantile(1 - pct)
    return int(series[labels == 1].ge(threshold).sum())


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def compare_feature_frames(recomputed: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    merged = recomputed.merge(scored[["member_id", *FEATURE_COLUMNS]], on="member_id", suffixes=("_recomputed", "_scored"))
    rows = []
    for column in FEATURE_COLUMNS:
        left = merged[f"{column}_recomputed"]
        right = merged[f"{column}_scored"]
        delta = (left - right).abs()
        rows.append(
            {
                "feature": column,
                "max_abs_diff": float(delta.max()),
                "mean_abs_diff": float(delta.mean()),
                "correlation": float(left.corr(right)) if left.nunique() > 1 and right.nunique() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["max_abs_diff", "mean_abs_diff"], ascending=False)


def build_recomputed_player_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_parquet(RAW_PATH)
    fraud_df = load_fraud_labels()
    normalized = normalize_member_history(raw_df)
    history_df, matched_fraud_rows = apply_pre_fraud_cutoff(normalized, fraud_df)
    player_df = aggregate_member_features(history_df)

    fraud_event_keys = set(fraud_df["fraud_event_key"])
    fraud_players = set(normalized.loc[normalized["fraud_event_key"].isin(fraud_event_keys), "member_id"])
    player_df["event_fraud_flag"] = player_df["member_id"].isin(fraud_players).astype(int)
    player_df["matched_fraud_rows_total"] = matched_fraud_rows
    return raw_df, history_df, player_df


def audit_data_quality(raw_df: pd.DataFrame, history_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
    print_header("Data Quality")
    timestamp_candidates = [
        "createdAt.$date",
        "createdat.$date",
        "trans_date.$date",
        "updatedAt.$date",
        "ts",
        "createdAt",
        "trans_date",
        "updatedAt",
    ]
    ts_col = next((column for column in timestamp_candidates if column in raw_df.columns), None)
    session_series = raw_df.get("session_id")
    session_numeric = pd.to_numeric(session_series, errors="coerce") if session_series is not None else pd.Series(dtype=float)
    non_numeric_sessions = int(session_series.notna().sum() - session_numeric.notna().sum()) if session_series is not None else 0
    zero_session_rows = int(session_numeric.fillna(0).eq(0).sum()) if session_series is not None else 0

    print(f"Raw rows: {len(raw_df):,}")
    print(f"History rows after cutoff: {len(history_df):,}")
    print(f"Player rows after aggregation: {len(player_df):,}")
    print(f"Fraud players after cutoff: {int(player_df['event_fraud_flag'].sum()):,}")
    print(f"Timestamp source column: {ts_col}")
    print(f"Rows with null normalized timestamps: {int(normalize_member_history(raw_df).ts.isna().sum()):,}")
    print(f"Rows with non-numeric session_id: {non_numeric_sessions:,}")
    print(f"Rows with session_id coerced to 0: {zero_session_rows:,}")

    positive_members = set(player_df.loc[player_df["event_fraud_flag"] == 1, "member_id"])
    normalized = normalize_member_history(raw_df)
    missing_ts_positive = int(normalized.loc[normalized["member_id"].isin(positive_members), "ts"].isna().sum())
    print(f"Positive-player rows with missing timestamps: {missing_ts_positive:,}")


def audit_scored_outputs(player_df: pd.DataFrame) -> None:
    print_header("Scored Output Consistency")
    scored = pd.read_parquet(SCORED_PATH).copy()
    scored["member_id"] = scored["member_id"].astype(str).str.upper().str.strip()

    print(f"Scored player rows: {len(scored):,}")
    print(f"Known fraud players in scored table: {int(scored['event_fraud_flag'].sum()):,}")
    print(f"Risk tier distribution: {scored['risk_tier'].value_counts().sort_index().to_dict()}")

    missing_from_scored = sorted(set(player_df["member_id"]) - set(scored["member_id"]))
    missing_from_recomputed = sorted(set(scored["member_id"]) - set(player_df["member_id"]))
    print(f"Members missing from scored table: {len(missing_from_scored)}")
    print(f"Members missing from recomputed table: {len(missing_from_recomputed)}")

    feature_comparison = compare_feature_frames(player_df, scored)
    worst = feature_comparison.head(10)
    print("Top feature drifts between recomputed features and scored table:")
    print(worst.to_string(index=False))

    score_formula_ok = np.allclose(
        scored["risk_score"],
        0.60 * scored["anomaly_score"] + 0.40 * scored["supervised_score"],
    )
    print(f"risk_score formula matches saved columns: {score_formula_ok}")

    if ARTIFACT_PATH.exists():
        artifacts = joblib.load(ARTIFACT_PATH)
        p80 = float(scored["risk_score"].quantile(0.80))
        p95 = float(scored["risk_score"].quantile(0.95))
        print(f"Artifact risk_p80 matches scored quantile: {np.isclose(artifacts['risk_p80'], p80)}")
        print(f"Artifact risk_p95 matches scored quantile: {np.isclose(artifacts['risk_p95'], p95)}")

    print("Top 10 scored players:")
    print(
        scored.sort_values("risk_score", ascending=False)[
            ["member_id", "risk_tier", "risk_score", "anomaly_score", "supervised_score", "event_fraud_flag"]
        ]
        .head(10)
        .to_string(index=False)
    )


def audit_eval_files() -> None:
    print_header("Evaluation File Consistency")
    scored = pd.read_parquet(SCORED_PATH)
    labels = scored["event_fraud_flag"].astype(int)

    with open(HYBRID_EVAL_PATH, "r", encoding="utf-8") as handle:
        hybrid_eval = json.load(handle)

    checks = {
        "anomaly": scored["anomaly_score"],
        "supervised_oos": scored["supervised_score_eval"],
        "combined_oos": scored["risk_score_eval"],
    }
    for method_name, series in checks.items():
        print(method_name)
        for pct_key, pct in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10), ("top_20pct", 0.20)]:
            actual = top_capture(series, labels, pct)
            saved = hybrid_eval["capture_rates"][method_name][pct_key]
            print(f"  {pct_key}: actual={actual}, saved={saved}, match={actual == saved}")


def audit_saved_feature_artifact(player_df: pd.DataFrame) -> None:
    print_header("Player Feature Artifact Consistency")
    saved = pd.read_parquet(PLAYER_FEATURE_PATH).copy()
    label_column = "player_label" if "player_label" in saved.columns else "event_fraud_flag"
    saved = saved.rename(columns={label_column: "event_fraud_flag"})
    print(f"Saved feature table rows: {len(saved):,}")
    print(f"Saved feature table fraud players: {int(saved['event_fraud_flag'].sum()):,}")
    print(f"Member set identical to recomputed: {set(saved['member_id']) == set(player_df['member_id'])}")


def main() -> None:
    raw_df, history_df, player_df = build_recomputed_player_frame()
    audit_data_quality(raw_df, history_df, player_df)
    audit_saved_feature_artifact(player_df)
    audit_scored_outputs(player_df)
    audit_eval_files()


if __name__ == "__main__":
    main()