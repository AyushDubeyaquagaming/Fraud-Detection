#!/usr/bin/env python
"""Classify fraud CSV rows against a raw parquet source."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.utils.fraud_label_matching import classify_fraud_csv, summarize_verdicts
from fraud_detection.utils.time_utils import normalize_event_timestamp


def _available_keys_from_parquet(parquet_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(parquet_path)
    if "member_id" not in raw.columns or "draw_id" not in raw.columns:
        raise ValueError("parquet source must contain member_id and draw_id columns")
    out = raw[[column for column in ["member_id", "draw_id"] if column in raw.columns]].copy()
    out["ts"] = normalize_event_timestamp(raw)
    return out


def classify(parquet_path: Path, fraud_csv_path: Path) -> tuple[pd.DataFrame, dict]:
    fraud_csv = pd.read_csv(fraud_csv_path)
    available_keys = _available_keys_from_parquet(parquet_path)
    verdicts = classify_fraud_csv(fraud_csv, available_keys)
    summary = {
        "parquet_path": str(parquet_path),
        "fraud_csv_path": str(fraud_csv_path),
        **summarize_verdicts(verdicts),
    }
    return verdicts, summary


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
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {len(out_df)} rows to {args.output}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
