#!/usr/bin/env python
"""Diagnostic: classify every fraud CSV entry against a raw parquet.

Answers WHY a set of expected fraud players may not survive the training-eval
feature-engineering pipeline, by bucketing each fraud CSV row into one of:
  - MATCHED                 fraud event found in parquet; member has pre-fraud history
  - DROPPED_NO_HISTORY      fraud event found; zero pre-fraud rows for that member
  - DROPPED_OUT_OF_WINDOW   member appears but fraud date is outside parquet date range
  - DROPPED_NOT_IN_PARQUET  member_id not present in parquet at all
  - DROPPED_KEY_MISMATCH    member present but no row with this exact draw_id

No production code is modified — outputs are an audit-only CSV + summary JSON.

Usage:
    python scripts/diagnose_fraud_label_matching.py \
        --parquet data_cache/fraud_modeling_pull.parquet \
        --fraud-csv "ROULET CHEATING DATA.csv" \
        --output diagnostic_fraud_label_matching.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.components.feature_engineering import _normalize_timestamp


def _normalize_parquet(parquet_path: Path) -> pd.DataFrame:
    """Load only the columns needed to match fraud events and compute histories."""
    raw = pd.read_parquet(
        parquet_path,
        columns=[
            c for c in [
                "member_id", "draw_id", "createdAt", "updatedAt", "trans_date",
            ] if c is not None
        ],
    )
    raw["member_id"] = raw["member_id"].astype(str).str.strip().str.upper()
    raw["draw_id"] = pd.to_numeric(raw["draw_id"], errors="coerce").astype("Int64")
    raw["ts"] = _normalize_timestamp(raw)
    return raw


def _normalize_fraud_csv(fraud_csv_path: Path) -> pd.DataFrame:
    fraud = pd.read_csv(fraud_csv_path)
    fraud.columns = [c.strip().lower() for c in fraud.columns]
    if "member_id" not in fraud.columns or "draw_id" not in fraud.columns:
        raise ValueError(
            f"fraud CSV must have member_id and draw_id columns; found {fraud.columns.tolist()}"
        )
    fraud["member_id_norm"] = fraud["member_id"].astype(str).str.strip().str.upper()
    fraud["draw_id_norm"] = pd.to_numeric(fraud["draw_id"], errors="coerce").astype("Int64")
    if "date" in fraud.columns:
        fraud["date_parsed"] = pd.to_datetime(fraud["date"], errors="coerce", utc=True)
    else:
        fraud["date_parsed"] = pd.NaT
    return fraud


def classify(parquet_path: Path, fraud_csv_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = _normalize_parquet(parquet_path)
    fraud = _normalize_fraud_csv(fraud_csv_path)

    parquet_min_ts = raw["ts"].min()
    parquet_max_ts = raw["ts"].max()

    members_in_parquet = set(raw["member_id"].dropna().unique())

    # Build (member, draw_id) key index once for the KEY_MISMATCH vs MATCHED decision
    raw_keys = pd.Series(
        raw["draw_id"].astype("Int64").astype("string") + "|" + raw["member_id"],
        name="event_key",
    )
    matched_event_keys = set(raw_keys.dropna().unique())

    # Per-member earliest fraud timestamp (for the cutoff-aware history counts)
    fraud_events_per_member = (
        fraud.groupby("member_id_norm")
        .agg(
            first_fraud_date=("date_parsed", "min"),
            fraud_event_count=("draw_id_norm", "size"),
        )
        .reset_index()
    )
    # Compute per-member (pre-fraud rows, post-fraud rows) using the per-member
    # earliest fraud date from the CSV (fallback: use the min ts of matched
    # (member, draw_id) pairs in the parquet when CSV date is missing).
    matched_pairs = fraud[["member_id_norm", "draw_id_norm"]].merge(
        raw[["member_id", "draw_id", "ts"]].rename(columns={"member_id": "member_id_norm", "draw_id": "draw_id_norm"}),
        on=["member_id_norm", "draw_id_norm"],
        how="inner",
    )
    matched_first_ts = (
        matched_pairs.groupby("member_id_norm")["ts"].min().rename("matched_first_ts")
    )
    fraud_events_per_member = fraud_events_per_member.merge(
        matched_first_ts, on="member_id_norm", how="left"
    )
    # Match the feature-engineering semantic: first_fraud_ts is the earliest
    # matched fraud-row timestamp (from the parquet). Fall back to the CSV's
    # first_fraud_date only if NO fraud row matches the parquet — in that case
    # no cutoff is ever applied by FE anyway (the member isn't in fraud_players_seen).
    fraud_events_per_member["effective_first_ts"] = fraud_events_per_member[
        "matched_first_ts"
    ].where(
        fraud_events_per_member["matched_first_ts"].notna(),
        fraud_events_per_member["first_fraud_date"],
    )

    # For each fraud CSV row, compute row counts and classification
    event_key_series = (
        fraud["draw_id_norm"].astype("string") + "|" + fraud["member_id_norm"]
    )
    fraud["matched_event_key"] = event_key_series.isin(matched_event_keys)
    fraud["found_in_parquet_member"] = fraud["member_id_norm"].isin(members_in_parquet)

    member_group = raw.groupby("member_id")
    per_member_rows = member_group.size().rename("rows_in_parquet")
    per_member_first_ts = member_group["ts"].min().rename("member_first_ts")
    per_member_last_ts = member_group["ts"].max().rename("member_last_ts")

    fraud = (
        fraud
        .merge(per_member_rows, left_on="member_id_norm", right_index=True, how="left")
        .merge(per_member_first_ts, left_on="member_id_norm", right_index=True, how="left")
        .merge(per_member_last_ts, left_on="member_id_norm", right_index=True, how="left")
        .merge(
            fraud_events_per_member[["member_id_norm", "effective_first_ts"]],
            on="member_id_norm",
            how="left",
        )
    )
    fraud.rename(
        columns={
            "member_first_ts": "first_row_ts",
            "member_last_ts": "last_row_ts",
            "effective_first_ts": "first_fraud_ts_computed",
        },
        inplace=True,
    )
    fraud["rows_in_parquet"] = fraud["rows_in_parquet"].fillna(0).astype(int)

    # Pre-fraud / post-fraud row counts (per fraud CSV row — the cutoff is by member,
    # so multiple fraud CSV rows of the same member repeat the same counts).
    def _counts_for_member(member_id: str, cutoff_ts):
        if pd.isna(cutoff_ts):
            return (0, 0)
        try:
            member_rows = member_group.get_group(member_id)
        except KeyError:
            return (0, 0)
        before = int((member_rows["ts"] < cutoff_ts).sum())
        after = int((member_rows["ts"] >= cutoff_ts).sum())
        return (before, after)

    pre_counts = []
    post_counts = []
    for _, row in fraud.iterrows():
        before, after = _counts_for_member(row["member_id_norm"], row["first_fraud_ts_computed"])
        pre_counts.append(before)
        post_counts.append(after)
    fraud["rows_before_cutoff"] = pre_counts
    fraud["rows_after_cutoff"] = post_counts

    def _status(row) -> str:
        if not row["found_in_parquet_member"]:
            return "DROPPED_NOT_IN_PARQUET"
        if not row["matched_event_key"]:
            # Member is present but this specific (member, draw_id) doesn't exist.
            # Distinguish "out of window" (all member rows outside fraud-date window)
            # from plain key mismatch.
            fraud_date = row.get("date_parsed", pd.NaT)
            first_row_ts = row["first_row_ts"]
            last_row_ts = row["last_row_ts"]
            if (
                pd.notna(fraud_date)
                and pd.notna(first_row_ts)
                and pd.notna(last_row_ts)
                and (fraud_date < first_row_ts or fraud_date > last_row_ts)
            ):
                return "DROPPED_OUT_OF_WINDOW"
            return "DROPPED_KEY_MISMATCH"
        # matched_event_key True → member has pre-fraud history iff rows_before_cutoff > 0
        if row["rows_before_cutoff"] > 0:
            return "MATCHED"
        return "DROPPED_NO_HISTORY"

    fraud["final_status"] = fraud.apply(_status, axis=1)

    out = pd.DataFrame({
        "fraud_csv_member_id": fraud["member_id_norm"],
        "fraud_csv_draw_id": fraud["draw_id_norm"],
        "fraud_csv_date": fraud.get("date"),
        "found_in_parquet": fraud["found_in_parquet_member"],
        "rows_in_parquet": fraud["rows_in_parquet"],
        "first_row_ts": fraud["first_row_ts"],
        "last_row_ts": fraud["last_row_ts"],
        "matched_event_key": fraud["matched_event_key"],
        "first_fraud_ts_computed": fraud["first_fraud_ts_computed"],
        "rows_before_cutoff": fraud["rows_before_cutoff"],
        "rows_after_cutoff": fraud["rows_after_cutoff"],
        "final_status": fraud["final_status"],
    })

    # Event-level summary
    event_level = out["final_status"].value_counts().to_dict()
    # Unique-player summary: for each member, worst-case outcome is "DROPPED_*"
    # only if NO fraud row for that member is MATCHED. Otherwise the player is
    # captured as a matched fraud player.
    per_member = out.groupby("fraud_csv_member_id")["final_status"].agg(list)

    def _collapse(statuses):
        if "MATCHED" in statuses:
            return "MATCHED"
        # Prefer more specific reasons over NOT_IN_PARQUET when a member has
        # mixed reasons across rows
        priority = [
            "DROPPED_NO_HISTORY",
            "DROPPED_OUT_OF_WINDOW",
            "DROPPED_KEY_MISMATCH",
            "DROPPED_NOT_IN_PARQUET",
        ]
        for p in priority:
            if p in statuses:
                return p
        return statuses[0]

    player_level = (
        per_member.apply(_collapse).value_counts().to_dict()
    )

    summary = {
        "parquet_path": str(parquet_path),
        "fraud_csv_path": str(fraud_csv_path),
        "parquet_ts_range": {
            "min": str(parquet_min_ts) if pd.notna(parquet_min_ts) else None,
            "max": str(parquet_max_ts) if pd.notna(parquet_max_ts) else None,
        },
        "fraud_csv_rows": int(len(fraud)),
        "fraud_csv_unique_members": int(fraud["member_id_norm"].nunique()),
        "parquet_rows": int(len(raw)),
        "parquet_unique_members": int(len(members_in_parquet)),
        "event_level_status_counts": {k: int(v) for k, v in event_level.items()},
        "player_level_status_counts": {k: int(v) for k, v in player_level.items()},
    }
    return out, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--fraud-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostic_fraud_label_matching.csv"))
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    out_df, summary = classify(args.parquet, args.fraud_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {len(out_df)} rows to {args.output}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
