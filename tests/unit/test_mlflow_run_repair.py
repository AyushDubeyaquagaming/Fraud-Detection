from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fraud_detection.utils.mlflow_run_repair import find_stale_running_runs, repair_stale_runs


class _FakeClient:
    def __init__(self, runs):
        self._runs = runs
        self.repaired: list[tuple[str, str]] = []

    def search_runs(self, experiment_ids, filter_string, max_results):
        assert experiment_ids == ["2"]
        assert "RUNNING" in filter_string
        return self._runs

    def set_terminated(self, run_id, status="KILLED"):
        self.repaired.append((run_id, status))


def _run(run_id: str, *, start_ms: int, status: str = "RUNNING", name: str = "run"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_ms, status=status),
        data=SimpleNamespace(tags={"mlflow.runName": name, "source": "training_pipeline"}),
    )


def test_find_stale_running_runs_filters_by_age():
    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    client = _FakeClient(
        [
            _run("old", start_ms=now_ms - int(5 * 3_600_000), name="old_run"),
            _run("fresh", start_ms=now_ms - int(15 * 60_000), name="fresh_run"),
        ]
    )

    stale = find_stale_running_runs(client, experiment_id="2", min_age_hours=1.0, now=now)

    assert [item["run_id"] for item in stale] == ["old"]
    assert stale[0]["run_name"] == "old_run"


def test_repair_stale_runs_is_dry_run_by_default():
    client = _FakeClient([])
    repaired = repair_stale_runs(client, [{"run_id": "old"}], execute=False)

    assert repaired == []
    assert client.repaired == []


def test_repair_stale_runs_marks_runs_killed_when_executed():
    client = _FakeClient([])

    repaired = repair_stale_runs(client, [{"run_id": "old"}, {"run_id": "older"}], execute=True, status="KILLED")

    assert repaired == ["old", "older"]
    assert client.repaired == [("old", "KILLED"), ("older", "KILLED")]