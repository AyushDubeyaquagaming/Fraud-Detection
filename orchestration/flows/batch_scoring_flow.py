"""Prefect flow wrapping the batch scoring pipeline.

Run locally (no Prefect server required):
    python orchestration/flows/batch_scoring_flow.py

Deploy to Prefect Cloud:
    prefect deploy --prefect-file orchestration/prefect.yaml
"""
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

from fraud_detection.constants.constants import CONFIG_FILE_PATH, BATCH_SCORING_REPORT_FILE
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
    def batch_scoring_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:
        logger = get_run_logger()
        logger.info("Batch scoring flow starting — config: %s", config_path)

        try:
            output_dir = BatchScoringPipeline(config_path=Path(config_path)).run()
        except Exception as exc:
            logger.error("Batch scoring failed: %s", exc)
            notify_failure(
                flow_name="fraud-detection-batch-scoring",
                error=str(exc),
            )
            raise

        report = _read_scoring_report(output_dir)
        scored = report.get(
            "scored_count",
            report.get("total_scored", report.get("total_players", "unknown")),
        )
        logger.info("Batch scoring complete — output_dir=%s scored=%s", output_dir, scored)
        return {
            "output_dir": str(output_dir),
            "scored_count": scored,
            "status": "FINISHED",
        }

else:
    def batch_scoring_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:  # type: ignore[misc]
        import logging
        logger = logging.getLogger("batch_scoring_flow")
        logger.info("Prefect not available — running batch scoring directly")

        output_dir = BatchScoringPipeline(config_path=Path(config_path)).run()
        report = _read_scoring_report(output_dir)
        return {
            "output_dir": str(output_dir),
            "scored_count": report.get(
                "scored_count",
                report.get("total_scored", report.get("total_players", "unknown")),
            ),
            "status": "FINISHED",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch scoring flow")
    parser.add_argument("--config", default=str(CONFIG_FILE_PATH))
    args = parser.parse_args()

    result = batch_scoring_flow(config_path=args.config)
    print(json.dumps(result, indent=2, default=str))
