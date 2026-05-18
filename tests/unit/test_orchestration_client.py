from __future__ import annotations

from datetime import date

import yaml

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


def test_widen_candidate_window_includes_label_date(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "partnership": {
                    "candidate_window": {
                        "start_date": "2026-05-04",
                        "end_date": "2026-05-11",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    start, end = orchestration_client.widen_candidate_window_for_label(
        labeled_draw_date="2026-05-13",
        config_path=config_path,
    )

    assert start == date(2026, 5, 4)
    assert end == date(2026, 5, 14)


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
