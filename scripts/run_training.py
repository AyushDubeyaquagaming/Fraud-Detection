#!/usr/bin/env python
"""CLI entry point for the full training pipeline.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from fraud_detection.constants.constants import CONFIG_FILE_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fraud detection training pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE_PATH,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    try:
        run_dir = TrainingPipeline(config_path=args.config).run()
        print(f"\nTraining pipeline completed successfully.")
        print(f"Run directory: {run_dir}")
        return 0
    except Exception as exc:
        print(f"\nTraining pipeline FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
