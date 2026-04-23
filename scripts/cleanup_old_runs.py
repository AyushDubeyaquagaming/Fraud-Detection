"""Artifact retention utility — removes old run directories.

Usage:
    # Dry run (default — no files deleted)
    python scripts/cleanup_old_runs.py

    # Keep the 5 most recent runs, delete the rest
    python scripts/cleanup_old_runs.py --keep 5 --execute

    # Custom artifact root
    python scripts/cleanup_old_runs.py --artifact-root artifacts --keep 10 --execute
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow script to run from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.constants.constants import ARTIFACT_ROOT
from fraud_detection.logger import get_logger

logger = get_logger(__name__)


def _sorted_run_dirs(runs_dir: Path) -> list[Path]:
    return sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )


def cleanup(artifact_root: Path, keep: int, execute: bool) -> None:
    runs_dir = artifact_root / "runs"
    current_dir = artifact_root / "current"

    if not runs_dir.exists():
        logger.info("No runs directory found at %s — nothing to clean", runs_dir)
        return

    run_dirs = _sorted_run_dirs(runs_dir)
    to_delete = run_dirs[:-keep] if keep > 0 else run_dirs

    if not to_delete:
        logger.info("Nothing to delete — %d run(s) present, keep=%d", len(run_dirs), keep)
        return

    logger.info(
        "Found %d run(s). Keeping %d most recent. Will %s %d run(s).",
        len(run_dirs),
        min(keep, len(run_dirs)),
        "DELETE" if execute else "DRY-RUN skip",
        len(to_delete),
    )

    for run_dir in to_delete:
        if execute:
            logger.info("Deleting: %s", run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)
        else:
            logger.info("[DRY RUN] Would delete: %s", run_dir)

    if not execute:
        logger.info("Dry run complete — pass --execute to actually delete.")

    # Verify artifacts/current is untouched
    if current_dir.exists():
        logger.info("artifacts/current is intact at %s", current_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up old training run artifacts.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_ROOT,
        help="Root artifacts directory (default: artifacts/)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of most-recent runs to retain (default: 5)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete directories (default: dry run)",
    )
    args = parser.parse_args()

    cleanup(
        artifact_root=args.artifact_root,
        keep=args.keep,
        execute=args.execute,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
