"""Bootstrap or update the Phase B rolling 90-day parquet store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.utils.rolling_parquet_store import bootstrap_rolling_window, update_rolling_window  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data_store" / "training_window")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--days-to-pull", type=int, default=7)
    parser.add_argument("--update", action="store_true", help="Append recent rows instead of bootstrapping from scratch.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    if args.update:
        result = update_rolling_window(
            output_root=args.output_root,
            lookback_days=args.lookback_days,
            days_to_pull=args.days_to_pull,
        )
    else:
        end_date = pd.Timestamp(args.end_date, tz="UTC") if args.end_date else pd.Timestamp.now(tz="UTC")
        start_date = pd.Timestamp(args.start_date, tz="UTC") if args.start_date else end_date - pd.Timedelta(days=args.lookback_days)
        result = bootstrap_rolling_window(start_date=start_date, end_date=end_date, output_root=args.output_root)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
