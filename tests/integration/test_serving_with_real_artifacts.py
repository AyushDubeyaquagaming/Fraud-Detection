"""
Integration test for the serving layer against real promoted artifacts.

Gated behind the environment variable RUN_SERVING_INTEGRATION=1 so it never
runs in CI unless explicitly enabled.

Usage:
    $env:RUN_SERVING_INTEGRATION = "1"
    pytest tests/integration/test_serving_with_real_artifacts.py -v
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SERVING_INTEGRATION") != "1",
    reason="Set RUN_SERVING_INTEGRATION=1 to run serving integration tests",
)


@pytest.fixture(scope="module")
def real_client():
    from fraud_detection.serving.app import create_app
    from fraud_detection.constants.constants import CURRENT_DIR, RUNS_DIR

    manifest_path = CURRENT_DIR / "serving_manifest.json"
    assert manifest_path.exists(), (
        f"No serving manifest at {manifest_path}. "
        "Run a successful training pipeline first."
    )

    with open(manifest_path) as f:
        manifest = json.load(f)

    app = create_app()
    with TestClient(app) as client:
        yield client, manifest


def test_health_with_real_artifacts(real_client):
    client, _ = real_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["artifacts_loaded"] is True


def test_model_info_matches_manifest(real_client):
    client, manifest = real_client
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["source_run_id"] == manifest["run_id"]
    assert body["total_scored_members"] > 0


def test_score_known_member(real_client):
    client, manifest = real_client
    from fraud_detection.constants.constants import CURRENT_DIR, HYBRID_SCORED_PLAYERS_FILE

    scored_path = CURRENT_DIR / HYBRID_SCORED_PLAYERS_FILE
    assert scored_path.exists(), (
        f"No weekly serving snapshot at {scored_path}. "
        "Run the training pipeline or batch scoring pipeline first."
    )
    df = pd.read_parquet(scored_path)
    member_id = str(df["member_id"].iloc[0]).strip().upper()

    r = client.get(f"/score/{member_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["member_id"] == member_id
    assert body["source_run_id"] == manifest["run_id"]
    assert body["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


def test_score_unknown_member_returns_insufficient_data(real_client):
    client, _ = real_client
    r = client.get("/score/MEMBER_THAT_DOES_NOT_EXIST_XYZ")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_data"
