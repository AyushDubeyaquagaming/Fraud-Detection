"""Prefect flow wrapping the training pipeline.

Run locally (no Prefect server required):
    python orchestration/flows/training_flow.py

Deploy to Prefect Cloud:
    prefect deploy --prefect-file orchestration/prefect.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure both the repo root and src/ are importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from prefect import flow, task, get_run_logger
    from prefect.context import get_run_context
    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False

from fraud_detection.constants.constants import CONFIG_FILE_PATH, RUN_METADATA_FILE
from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from orchestration.notifications import notify_failure


def _read_run_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / RUN_METADATA_FILE
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"run_dir": str(run_dir), "status": "UNKNOWN"}


if _PREFECT_AVAILABLE:
    @flow(name="fraud-detection-training", log_prints=True)
    def training_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:
        logger = get_run_logger()
        logger.info("Training flow starting — config: %s", config_path)

        try:
            run_dir = TrainingPipeline(config_path=Path(config_path)).run()
        except Exception as exc:
            logger.error("Training pipeline failed: %s", exc)
            notify_failure(
                flow_name="fraud-detection-training",
                error=str(exc),
            )
            raise

        meta = _read_run_metadata(run_dir)
        logger.info(
            "Training complete — run_id=%s promoted=%s",
            meta.get("run_id", "unknown"),
            meta.get("promoted", False),
        )
        return {
            "run_id": meta.get("run_id"),
            "promoted": meta.get("promoted", False),
            "run_dir": str(run_dir),
            "status": meta.get("status"),
            "completed_at": meta.get("completed_at"),
        }

else:
    def training_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:  # type: ignore[misc]
        """Fallback when Prefect is not installed — runs the pipeline directly."""
        import logging
        logger = logging.getLogger("training_flow")
        logger.info("Prefect not available — running pipeline directly")

        run_dir = TrainingPipeline(config_path=Path(config_path)).run()
        meta = _read_run_metadata(run_dir)
        return {
            "run_id": meta.get("run_id"),
            "promoted": meta.get("promoted", False),
            "run_dir": str(run_dir),
            "status": meta.get("status"),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training flow")
    parser.add_argument("--config", default=str(CONFIG_FILE_PATH))
    args = parser.parse_args()

    result = training_flow(config_path=args.config)
    print(json.dumps(result, indent=2, default=str))
