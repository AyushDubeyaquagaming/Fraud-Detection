#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
from mlflow.tracking import MlflowClient

from fraud_detection.constants.constants import MLFLOW_EXPERIMENT_NAME
from fraud_detection.utils.mlflow_run_repair import find_stale_running_runs, repair_stale_runs
from fraud_detection.utils.mlflow_utils import get_tracking_uri


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and optionally repair stale RUNNING MLflow runs.")
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--min-age-hours", type=float, default=1.0)
    parser.add_argument("--status", default="KILLED")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        print(json.dumps({"tracking_uri": tracking_uri, "experiment_name": args.experiment_name, "runs": []}, indent=2))
        return 0

    client = MlflowClient()
    runs = find_stale_running_runs(
        client,
        experiment_id=experiment.experiment_id,
        min_age_hours=float(args.min_age_hours),
    )
    repaired = repair_stale_runs(client, runs, execute=bool(args.execute), status=str(args.status))
    print(
        json.dumps(
            {
                "tracking_uri": tracking_uri,
                "experiment_id": experiment.experiment_id,
                "experiment_name": args.experiment_name,
                "execute": bool(args.execute),
                "requested_status": str(args.status),
                "stale_running_runs": runs,
                "repaired_run_ids": repaired,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())