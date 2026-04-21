#!/usr/bin/env python
"""Diagnostic: per-feature univariate separation between fraud and non-fraud.

For each numeric feature in `player_features.parquet`, compute:
  - mean/median for fraud players
  - mean/median for non-fraud players
  - KS statistic between the two distributions
  - AUC of that feature as a univariate classifier (higher = better separator)

Output: CSV sorted by univariate AUC descending.

Usage:
    python scripts/diagnose_feature_separation.py \
        --player-features artifacts/runs/<run_id>/feature_engineering/player_features.parquet \
        --output diagnostic_feature_separation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


EXCLUDE_COLS = {
    "member_id", "event_fraud_flag", "primary_ccs_id",
    "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player",
    "event_label", "fraud_event_key",
}


def compute(player_df: pd.DataFrame) -> pd.DataFrame:
    if "event_fraud_flag" not in player_df.columns:
        raise ValueError("player_features must include event_fraud_flag")
    labels = player_df["event_fraud_flag"].astype(int).to_numpy()
    fraud_mask = labels == 1
    non_fraud_mask = ~fraud_mask

    rows = []
    for col in player_df.columns:
        if col in EXCLUDE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(player_df[col]):
            continue
        values = player_df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        fraud_vals = values[fraud_mask].dropna()
        non_fraud_vals = values[non_fraud_mask].dropna()
        if len(fraud_vals) == 0 or len(non_fraud_vals) == 0:
            continue

        ks_stat = ks_2samp(fraud_vals, non_fraud_vals).statistic
        try:
            # Use feature as univariate score. If the fraud distribution tends lower,
            # invert; we take max(auc, 1 - auc) so the reported value always measures
            # absolute separation regardless of direction.
            valid = values.notna()
            auc_raw = roc_auc_score(labels[valid.to_numpy()], values[valid].to_numpy())
            auc = max(auc_raw, 1 - auc_raw)
        except ValueError:
            auc = float("nan")

        rows.append({
            "feature": col,
            "fraud_mean": float(fraud_vals.mean()),
            "non_fraud_mean": float(non_fraud_vals.mean()),
            "fraud_median": float(fraud_vals.median()),
            "non_fraud_median": float(non_fraud_vals.median()),
            "ks_stat": float(ks_stat),
            "univariate_auc": float(auc),
            "fraud_n": int(len(fraud_vals)),
            "non_fraud_n": int(len(non_fraud_vals)),
        })

    df = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    return df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostic_feature_separation.csv"))
    parser.add_argument("--top", type=int, default=15, help="print top-N rows")
    args = parser.parse_args()

    player_df = pd.read_parquet(args.player_features)
    result = compute(player_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Wrote {len(result)} feature rows to {args.output}")
    print(f"\nTop {args.top} features by univariate AUC:\n")
    print(result.head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
