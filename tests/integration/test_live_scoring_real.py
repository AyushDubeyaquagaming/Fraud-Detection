from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.serving.app import create_app
from fraud_detection.serving.artifact_provider import LocalDiskArtifactProvider


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCORING_INTEGRATION") != "1",
    reason="Requires real promoted artifacts and Mongo access",
)
def test_live_endpoint_against_real_member():
    provider = LocalDiskArtifactProvider(current_dir="artifacts/current")
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as client:
        scored = pd.read_parquet("artifacts/current/hybrid_scored_players.parquet")
        member_id = str(scored.iloc[0]["member_id"])
        response = client.get(f"/score/live/{member_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["member_id"] == member_id
    assert body["risk_score"] >= 0.0


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCORING_INTEGRATION") != "1",
    reason="Requires real promoted artifacts and Mongo access",
)
def test_historical_endpoint_against_real_member():
    provider = LocalDiskArtifactProvider(current_dir="artifacts/current")
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as client:
        scored = pd.read_parquet("artifacts/current/hybrid_scored_players.parquet")
        member_id = str(scored.iloc[0]["member_id"])
        response = client.post(
            "/score/historical",
            json={
                "member_id": member_id,
                "start_date": "2026-04-01T00:00:00Z",
                "end_date": "2026-04-08T00:00:00Z",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["member_id"] == member_id
