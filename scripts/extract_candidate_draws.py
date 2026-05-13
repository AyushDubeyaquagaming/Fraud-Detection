from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fraud_detection.extraction.candidate_extractor import (  # noqa: E402
    CandidateDrawExtractor,
    load_candidate_extraction_config,
    parse_utc_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract weekly candidate roulette draws from MongoDB into a partitioned parquet store."
    )
    parser.add_argument("--config", default="configs/candidate_extraction.yaml", help="Path to extraction YAML config.")
    parser.add_argument("--start-date", required=True, help="Inclusive UTC start date, e.g. 2026-04-25.")
    parser.add_argument("--end-date", required=True, help="Exclusive UTC end date, e.g. 2026-04-26.")
    parser.add_argument("--force", action="store_true", help="Re-extract partitions even when parquet exists.")
    parser.add_argument("--dry-run", action="store_true", help="Log chunks and paths without connecting or writing.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only the first N weekly chunks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_candidate_extraction_config(args.config)
    logging.getLogger("fraud_detection").setLevel(getattr(logging, config.runtime.log_level.upper(), logging.INFO))

    extractor = CandidateDrawExtractor(config)
    summary = extractor.run(
        start_date=parse_utc_date(args.start_date),
        end_date=parse_utc_date(args.end_date),
        force=args.force,
        dry_run=args.dry_run,
        limit_chunks=args.limit_chunks,
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

