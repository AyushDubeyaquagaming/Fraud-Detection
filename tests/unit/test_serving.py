from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.serving.app import create_app
from fraud_detection.serving.artifact_provider import ArtifactBundle, ArtifactProvider, LocalDiskArtifactProvider


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
        snapshot_metadata={
            "snapshot_type": "weekly_serving",
            "source_run_id": "run_test",
            "model_version": "hybrid_v1",
        },
        promotion_metadata={},
        evaluation_metadata={
            "scored_at": "2026-04-23T00:00:00+00:00",
            "anomaly_weight": 0.6,
            "supervised_weight": 0.4,
            "total_players": 3,
            "risk_tier_distribution": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
        },
        run_metadata={"run_id": "run_test", "status": "FINISHED"},
        snapshot_available=True,
        snapshot_reason=None,
        loaded_at=datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc),
        source_run_id="run_test",
        promoted_at="2026-04-22T00:00:00+00:00",
        evaluated_at="2026-04-23T00:00:00+00:00",
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
            snapshot_metadata={**b.snapshot_metadata, "source_run_id": run_id},
            promotion_metadata=b.promotion_metadata,
            evaluation_metadata=b.evaluation_metadata,
            run_metadata=b.run_metadata,
            snapshot_available=b.snapshot_available,
            snapshot_reason=b.snapshot_reason,
            loaded_at=b.loaded_at,
            source_run_id=run_id,
            promoted_at=b.promoted_at,
            evaluated_at=b.evaluated_at,
            model_version=b.model_version,
        )


class _SnapshotMissingProvider(ArtifactProvider):
    def is_available(self) -> bool:
        return True

    def load(self) -> ArtifactBundle:
        b = _make_bundle()
        return ArtifactBundle(
            scored_players_df=b.scored_players_df.iloc[0:0],
            serving_manifest=b.serving_manifest,
            snapshot_metadata={
                "snapshot_type": "weekly_serving",
                "source_run_id": "run_test",
                "model_version": "hybrid_v1",
                "snapshot_status": "insufficient_data",
                "lookback_days": 7,
            },
            promotion_metadata=b.promotion_metadata,
            evaluation_metadata={
                "scored_at": None,
                "anomaly_weight": 0.6,
                "supervised_weight": 0.4,
                "total_players": 0,
                "risk_tier_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
                "snapshot_status": "insufficient_data",
                "snapshot_reason": "Weekly serving snapshot has not been generated yet.",
            },
            run_metadata=b.run_metadata,
            snapshot_available=False,
            snapshot_reason="Weekly serving snapshot has not been generated yet.",
            loaded_at=b.loaded_at,
            source_run_id=b.source_run_id,
            promoted_at=b.promoted_at,
            evaluated_at=None,
            model_version=b.model_version,
        )


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


def test_score_not_found_returns_insufficient_data_payload():
    with _client(_GoodProvider()) as c:
        r = c.get("/score/GK999")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_data"
    assert body["member_id"] == "GK999"
    assert "not enough weekly data" in body["detail"].lower()


def test_score_when_artifacts_absent_returns_503():
    with _client(_FailProvider()) as c:
        r = c.get("/score/GK001")
    assert r.status_code == 503


def test_score_when_snapshot_missing_returns_insufficient_data_payload():
    with _client(_SnapshotMissingProvider()) as c:
        r = c.get("/score/GK001")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_data"
    assert body["member_id"] == "GK001"
    assert "snapshot has not been generated" in body["detail"].lower()


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
    assert body["snapshot_available"] is True
    assert body["snapshot_status"] == "ready"
    tiers = body["tier_distribution"]
    assert tiers["HIGH"] == 1
    assert tiers["MEDIUM"] == 1
    assert tiers["LOW"] == 1
    assert body["total_scored_members"] == 3
    assert body["anomaly_weight"] == pytest.approx(0.6)
    assert body["supervised_weight"] == pytest.approx(0.4)


def test_model_info_reports_missing_snapshot_state():
    with _client(_SnapshotMissingProvider()) as c:
        r = c.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["snapshot_available"] is False
    assert body["snapshot_status"] == "insufficient_data"
    assert body["snapshot_lookback_days"] == 7
    assert "snapshot has not been generated" in body["snapshot_reason"].lower()
    assert body["total_scored_members"] == 0


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


def test_local_disk_provider_requires_weekly_snapshot_for_scores(tmp_path):
    current_dir = tmp_path / "current"
    run_dir = tmp_path / "runs" / "run_test"
    current_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    (current_dir / "serving_manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "run_id": "run_test",
                "model_version": "hybrid_v1",
                "promoted_at": "2026-04-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_id": "run_test", "status": "FINISHED"}),
        encoding="utf-8",
    )

    bundle = LocalDiskArtifactProvider(current_dir=current_dir, repo_root=tmp_path).load()

    assert bundle.snapshot_available is False
    assert bundle.scored_players_df.empty
    assert bundle.evaluation_metadata["total_players"] == 0
    assert "not been generated" in bundle.snapshot_reason.lower()


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
