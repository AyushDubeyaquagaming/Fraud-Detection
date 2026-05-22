from __future__ import annotations

from fraud_detection import orchestration_client


class _FakeResponse:
    def __init__(self, payload: dict, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self) -> dict:
        return self._payload


def test_trigger_full_cycle_flow_uses_full_cycle_deployment(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []

    def fake_get(url: str, timeout: int):
        calls.append(("GET", url, None))
        assert url.endswith("/deployments/name/fraud-detection-full-cycle/full-cycle-weekly")
        return _FakeResponse({"id": "deployment-1"})

    def fake_post(url: str, json: dict, timeout: int):
        calls.append(("POST", url, json))
        assert url.endswith("/deployments/deployment-1/create_flow_run")
        assert json["parameters"]["start_date"] == "2026-05-04"
        assert json["parameters"]["end_date"] == "2026-05-11"
        return _FakeResponse(
            {
                "id": "flow-run-1",
                "name": "demo-run",
                "state": {"type": "PENDING", "name": "Pending"},
            }
        )

    monkeypatch.setenv("PREFECT_API_URL", "http://prefect:4200/api")
    monkeypatch.setattr(orchestration_client.requests, "get", fake_get)
    monkeypatch.setattr(orchestration_client.requests, "post", fake_post)

    run = orchestration_client.trigger_full_cycle_flow(start_date="2026-05-04", end_date="2026-05-11")

    assert run.flow_run_id == "flow-run-1"
    assert run.state_type == "PENDING"
    assert run.ui_url == "http://prefect:4200/flow-runs/flow-run/flow-run-1"
    assert [method for method, _url, _payload in calls] == ["GET", "POST"]


def test_trigger_full_cycle_flow_can_use_deployment_default_rolling_window(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []

    def fake_get(url: str, timeout: int):
        calls.append(("GET", url, None))
        return _FakeResponse({"id": "deployment-1"})

    def fake_post(url: str, json: dict, timeout: int):
        calls.append(("POST", url, json))
        assert "start_date" not in json["parameters"]
        assert "end_date" not in json["parameters"]
        return _FakeResponse(
            {
                "id": "flow-run-2",
                "name": "demo-run",
                "state": {"type": "PENDING", "name": "Pending"},
            }
        )

    monkeypatch.setenv("PREFECT_API_URL", "http://prefect:4200/api")
    monkeypatch.setattr(orchestration_client.requests, "get", fake_get)
    monkeypatch.setattr(orchestration_client.requests, "post", fake_post)

    run = orchestration_client.trigger_full_cycle_flow()

    assert run.flow_run_id == "flow-run-2"
    assert [method for method, _url, _payload in calls] == ["GET", "POST"]
