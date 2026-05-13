from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def find_stale_running_runs(
    client: Any,
    *,
    experiment_id: str,
    min_age_hours: float,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now_ts = now or datetime.now(timezone.utc)
    now_ms = int(now_ts.timestamp() * 1000)
    runs = client.search_runs([experiment_id], "attributes.status = 'RUNNING'", max_results=50_000)
    stale: list[dict[str, Any]] = []
    for run in runs:
        start_time = getattr(run.info, "start_time", None)
        if start_time is None:
            continue
        age_hours = (now_ms - int(start_time)) / 3_600_000.0
        if age_hours < float(min_age_hours):
            continue
        stale.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "source": run.data.tags.get("source"),
                "age_hours": age_hours,
                "status": run.info.status,
            }
        )
    stale.sort(key=lambda item: item["age_hours"], reverse=True)
    return stale


def repair_stale_runs(
    client: Any,
    runs: list[dict[str, Any]],
    *,
    execute: bool,
    status: str = "KILLED",
) -> list[str]:
    repaired: list[str] = []
    if not execute:
        return repaired
    for item in runs:
        client.set_terminated(item["run_id"], status=status)
        repaired.append(item["run_id"])
    return repaired