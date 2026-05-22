"""Validate a batch of pending fraud member_ids against MongoDB.

For each (member_id, draw_id, fraud_date) in configs/pending_fraud_ids.yaml,
pulls all transactions for that member from Mongo via the member_list
ingestion strategy, then reports per-member:

  - found_in_mongo: any transactions exist
  - transaction_count_total / in_90d_window
  - date_range
  - draw_id_present: did the specific draw_id show up in the player's history
  - ccs_id_match: did the captured CCS appear in the player's transactions
  - pre_fraud_draw_count: draws in [fraud_date - window_days, fraud_date)
  - verdict: "add", "needs_more_history", or "not_found"

Run::

    python scripts/validate_fraud_ids.py
    python scripts/validate_fraud_ids.py --pending-file configs/pending_fraud_ids.yaml --window-days 7

Outputs::

    artifacts/fraud_id_validation/<timestamp>/report.json
    artifacts/fraud_id_validation/<timestamp>/report.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.logger import get_logger  # noqa: E402
from fraud_detection.utils.mongodb import (  # noqa: E402
    build_query_batches_from_strategy,
    pull_query_batches_to_dataframe,
)

logger = get_logger(__name__)

DEFAULT_PENDING_PATH = REPO_ROOT / "configs" / "pending_fraud_ids.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "fraud_id_validation"


def _load_pending(pending_path: Path) -> list[dict]:
    """Flatten the YAML structure to one row per (draw_id, member_id, ccs_id)."""
    with open(pending_path) as f:
        config = yaml.safe_load(f)
    flat: list[dict] = []
    for event in config.get("events", []):
        draw_id = event["draw_id"]
        fraud_date = pd.to_datetime(event["fraud_date"], utc=True)
        for member in event.get("members", []):
            flat.append({
                "member_id": str(member["member_id"]).strip().upper(),
                "ccs_id": str(member["ccs_id"]).strip(),
                "draw_id": int(draw_id),
                "fraud_date": fraud_date,
            })
    if not flat:
        raise ValueError(f"No events parsed from {pending_path}")
    return flat


def _normalize_ccs_id(value: object) -> str:
    """Normalize CCS IDs so config values like 38767 match Mongo CCS038767."""
    raw = str(value).strip().upper()
    if not raw:
        return ""
    without_prefix = re.sub(r"^CCS[-_\s]*", "", raw)
    digits = re.sub(r"\D", "", without_prefix)
    if digits:
        return digits.lstrip("0") or "0"
    return without_prefix.strip()


def _pull_transactions(member_ids: list[str]) -> pd.DataFrame:
    """Pull all transactions for the given members from Mongo."""
    query_filters = build_query_batches_from_strategy(
        "member_list",
        {
            "member_ids_source": "inline",
            "member_ids": member_ids,
        },
    )
    df = pull_query_batches_to_dataframe(
        uri_env_var="MONGODB_URI",
        db_env_var="MONGODB_DATABASE",
        collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        query_filters=query_filters,
    )
    if df.empty:
        return df
    df["member_id_norm"] = df["member_id"].astype(str).str.strip().str.upper()
    if "trans_date" in df.columns:
        df["ts"] = pd.to_datetime(df["trans_date"], errors="coerce", utc=True)
    elif "createdAt" in df.columns:
        df["ts"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    else:
        df["ts"] = pd.NaT
    if "draw_id" in df.columns:
        df["draw_id_norm"] = pd.to_numeric(df["draw_id"], errors="coerce").astype("Int64")
    else:
        df["draw_id_norm"] = pd.array([pd.NA] * len(df), dtype="Int64")
    return df


def _verdict(row: dict, min_pre_fraud_draws: int) -> str:
    if not row["found_in_mongo"]:
        return "not_found"
    if not row["draw_id_present"] or row["ccs_id_match"] is not True:
        return "needs_review"
    if row["pre_fraud_draw_count"] < min_pre_fraud_draws:
        return "needs_more_history"
    return "add"


def _build_report(
    pending: list[dict],
    transactions: pd.DataFrame,
    window_days: int,
    today: pd.Timestamp,
    lookback_days: int,
    min_pre_fraud_draws: int,
) -> list[dict]:
    """Per-pending-row verdict including counts, date range, and match flags."""
    window = pd.Timedelta(days=window_days)
    lookback = pd.Timedelta(days=lookback_days)
    report = []
    for entry in pending:
        member_id = entry["member_id"]
        draw_id = entry["draw_id"]
        ccs_id = entry["ccs_id"]
        ccs_id_norm = _normalize_ccs_id(ccs_id)
        fraud_date = entry["fraud_date"]

        member_txns = (
            transactions.loc[transactions["member_id_norm"] == member_id]
            if not transactions.empty else transactions
        )
        found = not member_txns.empty
        in_90d = (
            int((member_txns["ts"] >= today - lookback).sum()) if found else 0
        )
        pre_fraud = (
            int(
                (
                    (member_txns["ts"] >= fraud_date - window)
                    & (member_txns["ts"] < fraud_date)
                ).sum()
            )
            if found else 0
        )
        ts_min = member_txns["ts"].min() if found else pd.NaT
        ts_max = member_txns["ts"].max() if found else pd.NaT
        draw_present = (
            bool((member_txns["draw_id_norm"] == draw_id).any()) if found else False
        )
        ccs_match: bool | None = None
        if found:
            for col in ("ccs_id", "primary_ccs_id"):
                if col in member_txns.columns:
                    ccs_match = bool(
                        member_txns[col].map(_normalize_ccs_id).eq(ccs_id_norm).any()
                    )
                    break

        row = {
            "member_id": member_id,
            "ccs_id": ccs_id,
            "ccs_id_normalized": ccs_id_norm,
            "draw_id": draw_id,
            "fraud_date": fraud_date.isoformat(),
            "found_in_mongo": found,
            "transaction_count_total": int(len(member_txns)) if found else 0,
            "transaction_count_in_90d_window": in_90d,
            "date_range_min": ts_min.isoformat() if pd.notna(ts_min) else None,
            "date_range_max": ts_max.isoformat() if pd.notna(ts_max) else None,
            "draw_id_present": draw_present,
            "ccs_id_match": ccs_match,
            "pre_fraud_draw_count": pre_fraud,
        }
        row["verdict"] = _verdict(row, min_pre_fraud_draws=min_pre_fraud_draws)
        report.append(row)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pending-file",
        type=Path,
        default=DEFAULT_PENDING_PATH,
        help="YAML file with pending fraud events (default: configs/pending_fraud_ids.yaml)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write the timestamped report directory",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Pre-fraud window size (days) used to count pre-fraud draws (default: 7)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Window for the 'in 90d' count (default: 90)",
    )
    parser.add_argument(
        "--min-pre-fraud-draws",
        type=int,
        default=1,
        help="Minimum pre-fraud draws for verdict='add' (default: 1)",
    )
    args = parser.parse_args()

    pending = _load_pending(args.pending_file)
    member_ids = sorted({row["member_id"] for row in pending})
    logger.info(
        "validate_fraud_ids: %d unique member(s) across %d pending row(s) — pulling from Mongo",
        len(member_ids), len(pending),
    )
    transactions = _pull_transactions(member_ids)
    logger.info(
        "validate_fraud_ids: pulled %d transactions for %d member(s)",
        len(transactions),
        transactions["member_id_norm"].nunique() if not transactions.empty else 0,
    )

    today = pd.Timestamp.now(tz="UTC")
    report = _build_report(
        pending=pending,
        transactions=transactions,
        window_days=args.window_days,
        today=today,
        lookback_days=args.lookback_days,
        min_pre_fraud_draws=args.min_pre_fraud_draws,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pending_file": str(args.pending_file),
        "window_days": args.window_days,
        "lookback_days": args.lookback_days,
        "min_pre_fraud_draws": args.min_pre_fraud_draws,
        "verdict_counts": {
            verdict: sum(1 for r in report if r["verdict"] == verdict)
            for verdict in ("add", "needs_more_history", "needs_review", "not_found")
        },
        "rows": report,
    }
    json_path = out_dir / "report.json"
    csv_path = out_dir / "report.csv"
    with open(json_path, "w") as f:
        json.dump(report_summary, f, indent=2)
    pd.DataFrame(report).to_csv(csv_path, index=False)

    logger.info("validate_fraud_ids: report written to %s", out_dir)
    print("\n=== Verdict summary ===")
    for verdict, count in report_summary["verdict_counts"].items():
        print(f"  {verdict}: {count}")
    print(f"\nFull report: {json_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
