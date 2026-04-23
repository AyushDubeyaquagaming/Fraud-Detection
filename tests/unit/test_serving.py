from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.serving.app import create_app
from fraud_detection.serving.artifact_provider import ArtifactBundle, ArtifactProvider


def _make_df() -> pd.DataFrame:
    rows = [
        {
            "member_id": "GK001",
            "risk_score": 1.08,
            "risk_tier": "HIGH",
            "anomaly_score": 1.22,
            "supervised_score": 0.74,
            "primary_ccs_id": "CCS_01",
        },
        {
            "member_id": "GK002",
            "risk_score": 0.30,
            "risk_tier": "LOW",
            "anomaly_score": 0.40,
            "supervised_score": 0.20,
            "primary_ccs_id": None,
        },
        {
            "member_id": "GK003",
            "risk_score": 0.60,
            "risk_tier": "MEDIUM",
            "anomaly_score": 0.50,
            "supervised_score": 0.50,
            "primary_ccs_id": "CCS_03",
        },
    ]
    df = pd.DataFrame(rows)
    df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
    return df.set_index("member_id", drop=False)


def _make_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        scored_players_df=_make_df(),
        serving_manifest={"run_id": "run_test", "model_version": "hybrid_v1", "promoted_at": "2026-04-22T00:00:00+00:00"},
        promotion_metadata={},
        evaluation_metadata={
            "evaluated_at": "2026-04-22T00:00:00+00:00",
            "anomaly_weight": 0.6,
            "supervised_weight": 0.4,
        },
        run_metadata={"run_id": "run_test", "status": "FINISHED"},
        loaded_at=datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc),
        source_run_id="run_test",
        promoted_at="2026-04-22T00:00:00+00:00",
        evaluated_at="2026-04-22T00:00:00+00:00",
        model_version="hybrid_v1",
    )


class _GoodProvider(ArtifactProvider):
    def is_available(self) -> bool:
        return True

    def load(self) -> ArtifactBundle:
        return _make_bundle()


class _FailProvider(ArtifactProvider):
    def is_available(self) -> bool:
        return False

    def load(self) -> ArtifactBundle:
        raise RuntimeError("Intentional load failure for tests")


class _ToggleProvider(ArtifactProvider):
    """First load succeeds; reload returns a different bundle."""

    def __init__(self):
        self._call = 0

    def is_available(self) -> bool:
        return True

    def load(self) -> ArtifactBundle:
        self._call += 1
        b = _make_bundle()
        run_id = f"run_reload_{self._call}"
        return ArtifactBundle(
            scored_players_df=b.scored_players_df,
            serving_manifest={**b.serving_manifest, "run_id": run_id},
            promotion_metadata=b.promotion_metadata,
            evaluation_metadata=b.evaluation_metadata,
            run_metadata=b.run_metadata,
            loaded_at=b.loaded_at,
            source_run_id=run_id,
            promoted_at=b.promoted_at,
            evaluated_at=b.evaluated_at,
            model_version=b.model_version,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(provider: ArtifactProvider) -> TestClient:
    return TestClient(create_app(provider=provider), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_when_loaded():
    with _client(_GoodProvider()) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["artifacts_loaded"] is True
    assert isinstance(body["uptime_seconds"], int)


def test_health_when_not_loaded():
    with _client(_FailProvider()) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["artifacts_loaded"] is False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_existing_member():
    with _client(_GoodProvider()) as c:
        r = c.get("/score/GK001")
    assert r.status_code == 200
    body = r.json()
    assert body["member_id"] == "GK001"
    assert body["risk_tier"] == "HIGH"
    assert body["risk_score"] == pytest.approx(1.08)
    assert body["anomaly_score"] == pytest.approx(1.22)
    assert body["ccs_id"] == "CCS_01"
    assert body["source_run_id"] == "run_test"
    assert body["model_version"] == "hybrid_v1"


def test_score_values_above_one_pass_schema():
    """risk_score and anomaly_score above 1.0 must not be rejected."""
    with _client(_GoodProvider()) as c:
        r = c.get("/score/GK001")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_score"] > 1.0
    assert body["anomaly_score"] > 1.0


def test_score_normalizes_member_id():
    """Lowercase or padded member_id is normalised to uppercase."""
    with _client(_GoodProvider()) as c:
        r = c.get("/score/gk001")
    assert r.status_code == 200
    assert r.json()["member_id"] == "GK001"


def test_score_not_found_returns_404():
    with _client(_GoodProvider()) as c:
        r = c.get("/score/GK999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_score_when_artifacts_absent_returns_503():
    with _client(_FailProvider()) as c:
        r = c.get("/score/GK001")
    assert r.status_code == 503


def test_score_null_ccs_id():
    """Members with no CCS id should return ccs_id: null."""
    with _client(_GoodProvider()) as c:
        r = c.get("/score/GK002")
    assert r.status_code == 200
    assert r.json()["ccs_id"] is None


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def test_model_info_returns_correct_tiers():
    with _client(_GoodProvider()) as c:
        r = c.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    tiers = body["tier_distribution"]
    assert tiers["HIGH"] == 1
    assert tiers["MEDIUM"] == 1
    assert tiers["LOW"] == 1
    assert body["total_scored_members"] == 3
    assert body["anomaly_weight"] == pytest.approx(0.6)
    assert body["supervised_weight"] == pytest.approx(0.4)


def test_root_returns_model_info_payload():
    with _client(_GoodProvider()) as c:
        r = c.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["source_run_id"] == "run_test"
    assert body["total_scored_members"] == 3


def test_openapi_uses_model_info_example_in_docs():
    with _client(_GoodProvider()) as c:
        schema = c.get("/openapi.json").json()

    model_info_schema = schema["components"]["schemas"]["ModelInfoResponse"]
    example = model_info_schema["example"]
    assert example["source_run_id"] == "run_20260422_105102"
    assert example["total_scored_members"] == 109708


def test_model_info_503_when_not_loaded():
    with _client(_FailProvider()) as c:
        r = c.get("/model-info")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Admin reload
# ---------------------------------------------------------------------------

def test_admin_reload_swaps_bundle():
    provider = _ToggleProvider()
    with _client(provider) as c:
        r1 = c.get("/model-info")
        assert r1.status_code == 200
        first_run = r1.json()["source_run_id"]

        r2 = c.post("/admin/reload")
        assert r2.status_code == 200
        body = r2.json()
        assert body["status"] == "reloaded"
        assert body["previous_run_id"] == first_run
        assert body["current_run_id"] != first_run

        r3 = c.get("/model-info")
        assert r3.json()["source_run_id"] == body["current_run_id"]


def test_admin_reload_failure_returns_503_without_raw_detail():
    provider = _GoodProvider()
    with _client(provider) as c:
        # Force provider to fail on reload by replacing its load method
        original_load = provider.load
        provider.load = lambda: (_ for _ in ()).throw(RuntimeError("disk error"))

        r = c.post("/admin/reload")
        assert r.status_code == 503
        detail = r.json()["detail"]
        # Must not leak raw exception text
        assert "disk error" not in detail
        assert "RuntimeError" not in detail

        provider.load = original_load
