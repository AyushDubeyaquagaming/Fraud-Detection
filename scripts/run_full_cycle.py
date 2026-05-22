#!/usr/bin/env python
"""CLI entry point for the full candidate-store training cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.constants.constants import CONFIG_FILE_PATH  # noqa: E402
from orchestration.flows.full_cycle_flow import run_full_cycle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate refresh, CCS refresh, training, promotion, and batch scoring.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE_PATH)
    parser.add_argument("--candidate-config", type=Path, default=Path("configs/candidate_extraction.yaml"))
    parser.add_argument("--ccs-config", type=Path, default=Path("configs/ccs_profit.yaml"))
    parser.add_argument("--batch-config", type=Path, default=None)
    parser.add_argument("--start-date", default=None, help="Inclusive candidate window start. Defaults to partnership.candidate_window.start_date.")
    parser.add_argument("--end-date", default=None, help="Exclusive candidate window end. Defaults to partnership.candidate_window.end_date.")
    parser.add_argument("--window-mode", choices=["fixed", "rolling"], default="rolling")
    parser.add_argument("--force-candidates", action="store_true", help="Re-extract candidate partitions even if they exist.")
    parser.add_argument("--force-ccs", action="store_true", help="Rebuild CCS profit days even if they exist.")
    args = parser.parse_args()

    try:
        result = run_full_cycle(
            config_path=args.config,
            candidate_config_path=args.candidate_config,
            ccs_config_path=args.ccs_config,
            batch_config_path=args.batch_config,
            start_date=args.start_date,
            end_date=args.end_date,
            window_mode=args.window_mode,
            force_candidates=args.force_candidates,
            force_ccs=args.force_ccs,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"\nFull cycle FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
