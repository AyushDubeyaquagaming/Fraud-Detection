from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

DEFAULT_PREFECT_API_URL = "http://localhost:4200/api"
FULL_CYCLE_DEPLOYMENT = "fraud-detection-full-cycle/full-cycle-weekly"


@dataclass(frozen=True)
class PrefectFlowRun:
    flow_run_id: str
    name: str | None
    state_type: str | None
    state_name: str | None
    ui_url: str


def prefect_api_url() -> str:
    return os.getenv("PREFECT_API_URL", DEFAULT_PREFECT_API_URL).rstrip("/")


def prefect_ui_url() -> str:
    configured_ui_url = os.getenv("PREFECT_UI_URL", "").strip()
    if configured_ui_url:
        return configured_ui_url.rstrip("/")
    api_url = prefect_api_url()
    return api_url[:-4] if api_url.endswith("/api") else api_url


def trigger_full_cycle_flow(
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    config_path: str = "configs/config.yaml",
    candidate_config_path: str = "configs/candidate_extraction.yaml",
    ccs_config_path: str = "configs/ccs_profit.yaml",
    deployment_name: str = FULL_CYCLE_DEPLOYMENT,
) -> PrefectFlowRun:
    deployment = _get_deployment_by_name(deployment_name)
    deployment_id = deployment["id"]
    parameters = {
        "config_path": config_path,
        "candidate_config_path": candidate_config_path,
        "ccs_config_path": ccs_config_path,
    }
    if bool(start_date) != bool(end_date):
        raise ValueError("start_date and end_date must be provided together")
    if start_date and end_date:
        parameters["start_date"] = _to_date(start_date).isoformat()
        parameters["end_date"] = _to_date(end_date).isoformat()
    response = requests.post(
        f"{prefect_api_url()}/deployments/{deployment_id}/create_flow_run",
        json={"parameters": parameters},
        timeout=20,
    )
    _raise_for_prefect(response)
    payload = response.json()
    return _flow_run_from_payload(payload)


def get_flow_run_status(flow_run_id: str) -> PrefectFlowRun:
    response = requests.get(f"{prefect_api_url()}/flow_runs/{flow_run_id}", timeout=20)
    _raise_for_prefect(response)
    return _flow_run_from_payload(response.json())


def _get_deployment_by_name(deployment_name: str) -> dict[str, Any]:
    if "/" not in deployment_name:
        raise ValueError("deployment_name must be formatted as '<flow_name>/<deployment_name>'")
    flow_name, name = deployment_name.split("/", 1)
    response = requests.get(f"{prefect_api_url()}/deployments/name/{flow_name}/{name}", timeout=20)
    if response.status_code == 404:
        raise RuntimeError(
            f"Prefect deployment {deployment_name!r} is not registered. "
            "Run: prefect deploy --prefect-file orchestration/prefect.yaml --all"
        )
    _raise_for_prefect(response)
    return response.json()


def _flow_run_from_payload(payload: dict[str, Any]) -> PrefectFlowRun:
    state = payload.get("state") or {}
    flow_run_id = str(payload["id"])
    return PrefectFlowRun(
        flow_run_id=flow_run_id,
        name=payload.get("name"),
        state_type=state.get("type"),
        state_name=state.get("name"),
        ui_url=f"{prefect_ui_url()}/flow-runs/flow-run/{flow_run_id}",
    )


def _raise_for_prefect(response: requests.Response) -> None:
    if response.ok:
        return
    raise RuntimeError(f"Prefect API error {response.status_code}: {response.text}")


def _to_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.date()
