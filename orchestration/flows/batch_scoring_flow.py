"""Prefect flow wrapping the partnership batch scoring pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from prefect import flow, get_run_logger

    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False

from fraud_detection.constants.constants import BATCH_SCORING_CONFIG_FILE_PATH, BATCH_SCORING_REPORT_FILE
from fraud_detection.pipeline.batch_scoring_pipeline import BatchScoringPipeline
from orchestration.notifications import notify_failure


def _read_scoring_report(output_dir: Path) -> dict:
    report_path = output_dir / BATCH_SCORING_REPORT_FILE
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {"output_dir": str(output_dir)}


if _PREFECT_AVAILABLE:

    @flow(name="fraud-detection-batch-scoring", log_prints=True)
    def batch_scoring_flow(config_path: str = str(BATCH_SCORING_CONFIG_FILE_PATH)) -> dict:
        logger = get_run_logger()
        logger.info("Batch scoring flow starting: config=%s", config_path)
        try:
            output_dir = BatchScoringPipeline(config_path=Path(config_path)).run()
        except Exception as exc:
            logger.error("Batch scoring failed: %s", exc)
            notify_failure(flow_name="fraud-detection-batch-scoring", error=str(exc))
            raise

        report = _read_scoring_report(output_dir)
        draws_scored = report.get("draws_scored", "unknown")
        logger.info("Batch scoring complete: output_dir=%s draws_scored=%s", output_dir, draws_scored)
        return {"output_dir": str(output_dir), "draws_scored": draws_scored, "status": "FINISHED"}

else:

    def batch_scoring_flow(config_path: str = str(BATCH_SCORING_CONFIG_FILE_PATH)) -> dict:  # type: ignore[misc]
        output_dir = BatchScoringPipeline(config_path=Path(config_path)).run()
        report = _read_scoring_report(output_dir)
        return {
            "output_dir": str(output_dir),
            "draws_scored": report.get("draws_scored", "unknown"),
            "status": "FINISHED",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run partnership batch scoring flow")
    parser.add_argument("--config", default=str(BATCH_SCORING_CONFIG_FILE_PATH))
    args = parser.parse_args()
    print(json.dumps(batch_scoring_flow(config_path=args.config), indent=2, default=str))
