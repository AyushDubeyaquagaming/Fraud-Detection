from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.components.feature_engineering import _normalize_raw_df, build_ccs_stats_lookup
from fraud_detection.components.model_training import FEATURE_COLUMNS
from fraud_detection.serving.app import create_app
from fraud_detection.serving.artifact_provider import ArtifactBundle, ArtifactProvider
from fraud_detection.serving.live_scoring.feature_builder import FeatureBuilder
from fraud_detection.serving.live_scoring.scorer import LiveScorer
from fraud_detection.serving.live_scoring.window_resolver import MongoWindowFetchError, WindowResolver


class _IdentityScaler:
    def transform(self, frame):
        return np.asarray(frame, dtype=float)


class _FakeIsoForest:
    def score_samples(self, X):
        return -np.asarray(X, dtype=float).sum(axis=1) / 100.0


class _FakeKMeans:
    def __init__(self, n_features: int):
        self.cluster_centers_ = np.zeros((1, n_features), dtype=float)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _FakeLR:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logits = X.sum(axis=1) / 100.0
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def _bets(amount: float) -> str:
    return json.dumps([{"number": "1", "bet_amount": amount}])


def _raw_rows() -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T00:00:00Z")
    rows = []
    for idx in range(8):
        rows.append(
            {
                "member_id": "GK001",
                "draw_id": idx + 1,
                "bets": _bets(10 + idx),
                "win_points": 5.0 + idx,
                "total_bet_amount": 10.0 + idx,
                "session_id": idx // 2,
                "ccs_id": "CCS1",
                "createdAt": base + pd.Timedelta(days=idx),
                "updatedAt": base + pd.Timedelta(days=idx),
                "trans_date": base + pd.Timedelta(days=idx),
            }
        )
    for idx in range(3):
        rows.append(
            {
                "member_id": "GK002",
                "draw_id": 100 + idx,
                "bets": _bets(20 + idx),
                "win_points": 4.0,
                "total_bet_amount": 20.0 + idx,
                "session_id": idx,
                "ccs_id": "CCS2",
                "createdAt": base + pd.Timedelta(days=idx),
                "updatedAt": base + pd.Timedelta(days=idx),
                "trans_date": base + pd.Timedelta(days=idx),
            }
        )
    return pd.DataFrame(rows)


def _fake_model_bundle(raw_df: pd.DataFrame) -> dict:
    normalized = _normalize_raw_df(raw_df)
    ccs_lookup = build_ccs_stats_lookup(normalized)
    n_features = len(FEATURE_COLUMNS)
    return {
        "iso_forest": _FakeIsoForest(),
        "kmeans": _FakeKMeans(n_features),
        "mean_vec": np.zeros(n_features, dtype=float),
        "cov_inv": np.eye(n_features, dtype=float),
        "scaler_unsup": _IdentityScaler(),
        "scaler_operational": _IdentityScaler(),
        "lr_operational": _FakeLR(),
        "feature_columns": FEATURE_COLUMNS,
        "log1p_columns": [],
        "anomaly_weight": 0.6,
        "supervised_weight": 0.4,
        "anomaly_component_weights": {
            "iso_forest_score_norm": 0.4,
            "mahalanobis_norm": 0.3,
            "cluster_distance_norm": 0.3,
        },
        "iso_min": 0.0,
        "iso_max": 100.0,
        "mahal_min": 0.0,
        "mahal_max": 1000.0,
        "cluster_min": 0.0,
        "cluster_max": 1000.0,
        "risk_p80": 0.5,
        "risk_p95": 0.8,
        "ccs_stats_lookup": ccs_lookup,
    }


def _make_bundle(tmp_path, raw_df: pd.DataFrame) -> ArtifactBundle:
    raw_path = tmp_path / "raw_data.parquet"
    raw_df.to_parquet(raw_path, index=False)
    scored_df = pd.DataFrame(columns=["member_id"]).set_index("member_id", drop=False)
    timestamps = pd.to_datetime(raw_df["trans_date"], utc=True)
    return ArtifactBundle(
        scored_players_df=scored_df,
        serving_manifest={"run_id": "run_live", "model_version": "hybrid_v1", "promoted_at": "2026-04-22T00:00:00+00:00"},
        snapshot_metadata={"source_run_id": "run_live", "model_version": "hybrid_v1", "timestamp_field": "trans_date", "lookback_days": 7},
        promotion_metadata={},
        evaluation_metadata={"scored_at": "2026-04-23T00:00:00+00:00", "anomaly_weight": 0.6, "supervised_weight": 0.4},
        run_metadata={"run_id": "run_live", "status": "FINISHED"},
        snapshot_available=False,
        snapshot_reason="Weekly snapshot not required for live endpoint tests.",
        loaded_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        source_run_id="run_live",
        promoted_at="2026-04-22T00:00:00+00:00",
        evaluated_at="2026-04-23T00:00:00+00:00",
        model_version="hybrid_v1",
        model_bundle=_fake_model_bundle(raw_df),
        training_raw_parquet_path=raw_path,
        training_parquet_start_date=timestamps.min().to_pydatetime(),
        training_parquet_end_date=timestamps.max().to_pydatetime(),
    )


class _Provider(ArtifactProvider):
    def __init__(self, bundle: ArtifactBundle):
        self.bundle = bundle

    def is_available(self) -> bool:
        return True

    def load(self) -> ArtifactBundle:
        return self.bundle


def _client(bundle: ArtifactBundle) -> TestClient:
    return TestClient(create_app(provider=_Provider(bundle)), raise_server_exceptions=False)


def test_window_entirely_in_parquet_pulls_only_parquet(tmp_path):
    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path, parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(), parquet_end_date=pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime())

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-05T00:00:00Z").to_pydatetime(),
    )
    assert data.mongo_rows == 0
    assert data.parquet_rows == 3
    assert data.data_sources == ["training_parquet"]


def test_window_extends_past_parquet_pulls_mongo_delta(tmp_path, monkeypatch):
    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path, parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(), parquet_end_date=pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime())

    post_df = pd.DataFrame(
        [
            {
                "member_id": "GK001",
                "draw_id": 999,
                "bets": _bets(30),
                "win_points": 5.0,
                "total_bet_amount": 30.0,
                "session_id": 1,
                "ccs_id": "CCS1",
                "trans_date": pd.Timestamp("2026-04-07T00:00:00Z"),
            }
        ]
    )
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: post_df)

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-05T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-08T00:00:00Z").to_pydatetime(),
    )
    assert data.parquet_rows >= 1
    assert data.mongo_rows == 1
    assert "mongo_delta" in data.data_sources


def test_window_entirely_after_parquet_pulls_only_mongo(tmp_path, monkeypatch):
    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path, parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(), parquet_end_date=pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime())
    mongo_df = pd.DataFrame([{"member_id": "GK001", "draw_id": 1000, "trans_date": pd.Timestamp("2026-04-07T00:00:00Z")}])
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: mongo_df)

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-07T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-08T00:00:00Z").to_pydatetime(),
    )
    assert data.parquet_rows == 0
    assert data.mongo_rows == 1
    assert data.data_sources == ["mongo_delta"]


def test_empty_window_returns_empty_df(tmp_path, monkeypatch):
    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path)
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: pd.DataFrame())
    data = resolver.resolve(
        "MISSING",
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-03T00:00:00Z").to_pydatetime(),
    )
    assert data.raw_df.empty


def test_deduplicates_draws_across_sources_on_member_id_and_draw_id_only(tmp_path, monkeypatch):
    raw_df = _raw_rows().iloc[[0]].copy()
    raw_df.loc[:, "member_id"] = "GK001"
    raw_df.loc[:, "draw_id"] = 1
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(
        raw_path,
        parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
        parquet_end_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
    )
    mongo_df = raw_df.copy()
    mongo_df.loc[:, "trans_date"] = pd.Timestamp("2026-04-01T00:00:00.123Z")
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: mongo_df)

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-03-31T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
    )
    assert len(data.raw_df) == 1


def test_parquet_filter_matches_lowercase_padded_member_ids(tmp_path, monkeypatch):
    raw_df = _raw_rows().iloc[[0]].copy()
    raw_df.loc[:, "member_id"] = "  gk001  "
    raw_path = tmp_path / "raw_norm.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path)
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: pd.DataFrame())

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
    )
    assert len(data.raw_df) == 1


def test_parquet_filter_uses_training_timestamp_fallback(tmp_path, monkeypatch):
    raw_df = _raw_rows().iloc[[0]].copy()
    raw_df.loc[:, "trans_date"] = pd.NaT
    raw_df.loc[:, "createdAt"] = pd.Timestamp("2026-04-01T00:00:00Z")
    raw_path = tmp_path / "raw_ts_fallback.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(raw_path, timestamp_field="trans_date")
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: pd.DataFrame())

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
    )
    assert len(data.raw_df) == 1


def test_window_including_exact_parquet_end_keeps_boundary_row(tmp_path, monkeypatch):
    raw_df = _raw_rows().iloc[[5]].copy()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    end_ts = pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime()
    resolver = WindowResolver(raw_path, parquet_start_date=end_ts, parquet_end_date=end_ts)
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: pd.DataFrame())

    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-05T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-07T00:00:00Z").to_pydatetime(),
    )
    assert len(data.raw_df) == 1
    assert int(data.raw_df.iloc[0]["draw_id"]) == 6


def test_feature_builder_produces_correct_columns():
    raw_df = _raw_rows()
    builder = FeatureBuilder(FEATURE_COLUMNS, _fake_model_bundle(raw_df)["ccs_stats_lookup"])
    feature_df = builder.build("GK001", raw_df[raw_df["member_id"] == "GK001"])
    assert list(feature_df.columns) == ["member_id"] + FEATURE_COLUMNS + ["primary_ccs_id"]


def test_feature_builder_returns_empty_for_empty_input():
    builder = FeatureBuilder(FEATURE_COLUMNS, pd.DataFrame())
    assert builder.build("GK001", pd.DataFrame()).empty


def test_scorer_produces_valid_risk_score_range():
    raw_df = _raw_rows()
    feature_df = FeatureBuilder(FEATURE_COLUMNS, _fake_model_bundle(raw_df)["ccs_stats_lookup"]).build(
        "GK001",
        raw_df[raw_df["member_id"] == "GK001"],
    )
    result = LiveScorer(_fake_model_bundle(raw_df)).score(feature_df)
    assert result.risk_score >= 0.0
    assert result.risk_tier in {"LOW", "MEDIUM", "HIGH"}


def test_scorer_raises_on_empty_feature_df():
    with pytest.raises(ValueError):
        LiveScorer(_fake_model_bundle(_raw_rows())).score(pd.DataFrame())


def test_live_endpoint_returns_valid_response(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, _raw_rows())

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr("fraud_detection.serving.routes.live_scoring.datetime", _FixedDateTime)
    with _client(bundle) as c:
        r = c.get("/score/live/gk001")
    assert r.status_code == 200
    body = r.json()
    assert body["member_id"] == "GK001"
    assert body["window_days"] == 7
    assert body["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}


def test_historical_endpoint_validates_date_order(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "GK001", "start_date": "2026-04-04T00:00:00Z", "end_date": "2026-04-03T00:00:00Z"})
    assert r.status_code == 400


def test_historical_endpoint_rejects_future_end_date(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "GK001", "start_date": "2026-04-01T00:00:00Z", "end_date": future})
    assert r.status_code == 400


def test_historical_endpoint_rejects_zero_day_range(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "GK001", "start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-01T12:00:00Z"})
    assert r.status_code == 400


def test_endpoint_returns_insufficient_data_for_unknown_member(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "UNKNOWN", "start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-03T00:00:00Z"})
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_data"


def test_endpoint_returns_503_when_artifacts_not_loaded(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    bundle = ArtifactBundle(
        **{**bundle.__dict__, "model_bundle": None}
    )
    with _client(bundle) as c:
        r = c.get("/score/live/GK001")
    assert r.status_code == 503


def test_endpoint_returns_503_when_bundle_lacks_ccs_lookup(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    model_bundle = dict(bundle.model_bundle)
    model_bundle.pop("ccs_stats_lookup", None)
    bundle = ArtifactBundle(
        **{**bundle.__dict__, "model_bundle": model_bundle}
    )
    with _client(bundle) as c:
        r = c.get("/score/live/GK001")
    assert r.status_code == 503


def test_endpoint_returns_504_when_mongo_is_required_but_unavailable(tmp_path, monkeypatch):
    raw_df = _raw_rows().iloc[:2].copy()
    bundle = _make_bundle(tmp_path, raw_df)

    def _boom(*args, **kwargs):
        raise MongoWindowFetchError("MongoDB connection failed: boom")

    monkeypatch.setattr("fraud_detection.serving.live_scoring.window_resolver.get_serving_mongo_collection", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "GK001", "start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-10T00:00:00Z"})
    assert r.status_code == 504


def test_endpoint_normalizes_member_id_case(tmp_path):
    bundle = _make_bundle(tmp_path, _raw_rows())
    with _client(bundle) as c:
        r = c.post("/score/historical", json={"member_id": "gk001", "start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-03T00:00:00Z"})
    assert r.status_code == 200
    assert r.json()["member_id"] == "GK001"


def test_live_path_never_calls_pd_read_parquet(tmp_path, monkeypatch):
    """Memory-safety guard: a single live request must not trigger a full
    `pd.read_parquet` of the training cohort. We replace `pd.read_parquet`
    with a sentinel that raises if called, then exercise the resolver end
    to end. The streaming pyarrow scanner should be the only path used.
    """
    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "pd.read_parquet must not be called from the live path — bounds "
            "should come from row-group metadata and per-request reads should "
            "stream via pyarrow.dataset.scanner."
        )

    monkeypatch.setattr("pandas.read_parquet", _boom)

    resolver = WindowResolver(raw_path)
    monkeypatch.setattr(resolver, "_pull_from_mongo", lambda *args, **kwargs: pd.DataFrame())
    data = resolver.resolve(
        "GK001",
        pd.Timestamp("2026-04-02T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-05T00:00:00Z").to_pydatetime(),
    )
    assert data.parquet_rows >= 1


def test_mongo_pull_uses_projection_and_batch_size(tmp_path, monkeypatch):
    """Mongo lean-read guard (Fix 4): the live path must call
    `collection.find(query, MONGO_PROJECTION).batch_size(...)` rather than
    `find(query)` — the latter pulls every document field, which is what
    drove the original RAM blow-up.
    """
    from fraud_detection.utils import mongodb as mongodb_module

    captured = {"projection": None, "batch_size": None}

    class _FakeCursor:
        def __init__(self):
            self._docs = []

        def batch_size(self, value):
            captured["batch_size"] = value
            return self

        def close(self):
            pass

        def __iter__(self):
            return iter(self._docs)

    class _FakeCollection:
        def find(self, query, projection=None):  # noqa: ARG002 - signature mirrors pymongo
            captured["projection"] = projection
            return _FakeCursor()

    monkeypatch.setattr(
        "fraud_detection.serving.live_scoring.window_resolver.get_serving_mongo_collection",
        lambda *args, **kwargs: _FakeCollection(),
    )

    raw_df = _raw_rows()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(
        raw_path,
        parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
        parquet_end_date=pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime(),
    )
    resolver.resolve(
        "GK001",
        pd.Timestamp("2026-03-25T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-08T00:00:00Z").to_pydatetime(),
    )

    assert captured["projection"] == mongodb_module.MONGO_PROJECTION
    assert captured["batch_size"] == 10_000


def test_mongo_pull_serializes_bson_like_bets_and_drops_id(tmp_path, monkeypatch):
    class _FakeObjectId:
        def __str__(self):
            return "fake-object-id"

    class _FakeCursor:
        def __init__(self, docs):
            self._docs = docs

        def batch_size(self, _value):
            return self

        def close(self):
            pass

        def __iter__(self):
            return iter(self._docs)

    class _FakeCollection:
        def find(self, _query, _projection=None):
            return _FakeCursor(
                [
                    {
                        "_id": _FakeObjectId(),
                        "member_id": "GK001",
                        "draw_id": 2001,
                        "bets": [{"number": "7", "bet_amount": 15.0, "bet_id": _FakeObjectId()}],
                        "win_points": 3.0,
                        "total_bet_amount": 15.0,
                        "session_id": 9,
                        "ccs_id": "CCS1",
                        "createdAt": pd.Timestamp("2026-04-07T00:00:00Z").to_pydatetime(),
                        "updatedAt": pd.Timestamp("2026-04-07T00:00:00Z").to_pydatetime(),
                        "trans_date": pd.Timestamp("2026-04-07T00:00:00Z").to_pydatetime(),
                    }
                ]
            )

    monkeypatch.setattr(
        "fraud_detection.serving.live_scoring.window_resolver.get_serving_mongo_collection",
        lambda *args, **kwargs: _FakeCollection(),
    )

    raw_df = _raw_rows().iloc[:1].copy()
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    resolver = WindowResolver(
        raw_path,
        parquet_start_date=pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime(),
        parquet_end_date=pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime(),
    )

    mongo_df = resolver._pull_from_mongo(
        "GK001",
        pd.Timestamp("2026-04-06T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2026-04-08T00:00:00Z").to_pydatetime(),
        start_inclusive=False,
    )

    assert len(mongo_df) == 1
    assert "_id" not in mongo_df.columns
    assert mongo_df.iloc[0]["bets"] == '[{"number": "7", "bet_amount": 15.0, "bet_id": "fake-object-id"}]'
