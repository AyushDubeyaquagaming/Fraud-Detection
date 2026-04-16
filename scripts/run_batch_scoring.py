#!/usr/bin/env python
"""CLI entry point for batch scoring pipeline.

Usage:
    python scripts/run_batch_scoring.py
    python scripts/run_batch_scoring.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.pipeline.batch_scoring_pipeline import BatchScoringPipeline
from fraud_detection.constants.constants import CONFIG_FILE_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fraud detection batch scoring pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE_PATH,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    try:
        output_dir = BatchScoringPipeline(config_path=args.config).run()
        print(f"\nBatch scoring completed successfully.")
        print(f"Outputs in: {output_dir}")
        return 0
    except Exception as exc:
        print(f"\nBatch scoring FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
