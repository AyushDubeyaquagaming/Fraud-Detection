from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from fraud_detection.utils import native_feedback


class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = docs or []

    def find(self, *_args, **_kwargs):
        return list(self.docs)


class _FakeResponse:
    status_code = 200
    text = "ok"


def test_extract_events_includes_flagged_and_partnership_members_with_ccs_context():
    doc = {
        "requires_review": True,
        "flagged_members": [{"member_id": "a"}],
        "partnerships": [{"member_ids": ["A", "B"]}],
    }
    row = {
        "member_ids": ["A", "B"],
        "ccs_ids": ["CA", "CB"],
        "trans_date_min": pd.Timestamp("2026-05-01T12:00:00Z"),
    }

    events = native_feedback.extract_native_feedback_events(doc, row)

    assert {(event.member_id, event.ccs_id, event.alert_date.isoformat()) for event in events} == {
        ("A", "CA", "2026-05-01"),
        ("B", "CB", "2026-05-01"),
    }


def test_group_member_alert_periods_collapses_consecutive_days():
    events = [
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-01").date()),
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-02").date()),
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-04").date()),
    ]

    periods = native_feedback.group_member_alert_periods(events)

    assert [(start.isoformat(), end.isoformat()) for start, end in periods["A"]] == [
        ("2026-05-01", "2026-05-02"),
        ("2026-05-04", "2026-05-04"),
    ]


def test_build_suspicious_bulk_payload_collapses_member_periods():
    events = [
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-01").date()),
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-02").date()),
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-04").date()),
    ]

    payload = native_feedback.build_suspicious_bulk_payload(events)

    assert payload == [
        {
            "memberId": "A",
            "fromDate": "2026-05-01T00:00:00.000Z",
            "toDate": "2026-05-02T00:00:00.000Z",
        },
        {
            "memberId": "A",
            "fromDate": "2026-05-04T00:00:00.000Z",
            "toDate": "2026-05-04T00:00:00.000Z",
        },
    ]


def test_sync_native_suspected_feedback_posts_bulk_payload(monkeypatch):
    calls = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        calls["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(native_feedback.requests, "post", fake_post)
    events = [
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-01").date()),
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-02").date()),
    ]

    summary = native_feedback.sync_native_suspected_feedback(
        events,
        {
            "enabled": True,
            "api_base_url": "https://backend.example/api",
            "suspicious_bulk_path": "/gk-users/suspicious/bulk",
            "request_timeout_seconds": 12,
        },
    )

    assert summary["status"] == "sent"
    assert summary["payloads"] == 1
    assert calls["url"] == "https://backend.example/api/gk-users/suspicious/bulk"
    assert calls["headers"] == {"Content-Type": "application/json"}
    assert calls["timeout"] == 12
    assert calls["json"] == [
        {
            "memberId": "A",
            "fromDate": "2026-05-01T00:00:00.000Z",
            "toDate": "2026-05-02T00:00:00.000Z",
        }
    ]


def test_sync_native_suspected_feedback_skips_missing_api_base_url(monkeypatch):
    def fail_post(*_args, **_kwargs):
        raise AssertionError("missing API base URL should not call backend")

    warnings = []
    monkeypatch.delenv("GK_BACKEND_API_BASE_URL", raising=False)
    monkeypatch.setattr(native_feedback.requests, "post", fail_post)
    monkeypatch.setattr(native_feedback.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    events = [
        native_feedback.NativeFeedbackEvent("A", "CA", pd.Timestamp("2026-05-01").date()),
    ]

    summary = native_feedback.sync_native_suspected_feedback(events, {"enabled": True})

    assert summary == {
        "enabled": True,
        "events": 1,
        "members": 1,
        "payloads": 1,
        "status": "skipped_missing_api_base_url",
    }
    assert "backend API base URL is missing" in warnings[0][0]


def test_read_native_feedback_labels_uses_user_periods_only(monkeypatch):
    users_collection = _FakeCollection(
        [
            {
                "member_id": "A",
                "confirmed_fraud": False,
                "suspected_fraud": [
                    {
                        "memberId": "A",
                        "fromDate": datetime(2026, 5, 1, tzinfo=timezone.utc),
                        "toDate": datetime(2026, 5, 1, tzinfo=timezone.utc),
                    }
                ],
            },
            {"member_id": "B", "confirmed_fraud": False, "suspected_fraud": []},
        ]
    )

    def fake_collection(_uri, _db, collection_env_var):
        assert collection_env_var == "MONGODB_COLLECTION_USERS"
        return users_collection

    monkeypatch.setattr(native_feedback, "get_serving_mongo_collection", fake_collection)
    rows = pd.DataFrame(
        [
            {
                "draw_id": 1,
                "member_ids": ["A", "B"],
                "ccs_ids": ["CA", "CB"],
                "trans_date_min": pd.Timestamp("2026-05-01T12:00:00Z"),
            }
        ]
    )

    labels = native_feedback.read_native_feedback_labels_for_candidate_rows(
        rows,
        {
            "enabled": True,
            "users_collection_env_var": "MONGODB_COLLECTION_USERS",
            "confirmed_not_fraud_weight": 3.0,
        },
    )

    by_member = {item["member_id"]: item["label"] for item in labels}
    assert by_member == {"A": "not_fraud"}
    assert labels[0]["sample_weight"] == 3.0


def test_read_native_feedback_labels_accepts_boolean_suspected_flag(monkeypatch):
    users_collection = _FakeCollection(
        [
            {"member_id": "A", "confirmed_fraud": True, "suspected_fraud": True},
        ]
    )

    monkeypatch.setattr(
        native_feedback,
        "get_serving_mongo_collection",
        lambda _uri, _db, collection_env_var: users_collection,
    )
    rows = pd.DataFrame(
        [
            {
                "draw_id": 1,
                "member_ids": ["A"],
                "ccs_ids": ["CA"],
                "trans_date_min": pd.Timestamp("2026-05-01T12:00:00Z"),
            }
        ]
    )

    labels = native_feedback.read_native_feedback_labels_for_candidate_rows(
        rows,
        {"enabled": True, "confirmed_fraud_weight": 5.0},
    )

    assert labels[0]["label"] == "fraud"
    assert labels[0]["sample_weight"] == 5.0
