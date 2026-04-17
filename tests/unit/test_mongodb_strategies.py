"""Unit tests for MongoDB strategy query builders.

These tests do not hit a real MongoDB instance — they only verify the
query-building logic in fraud_detection.utils.mongodb.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.exception import FraudDetectionException
from fraud_detection.utils.mongodb import (
    build_query_batches_from_strategy,
    _build_date_window_queries,
    _build_member_list_queries,
    _resolve_member_ids,
    _validate_full_pull_confirmed,
)


# ---------------------------------------------------------------------------
# date_window strategy
# ---------------------------------------------------------------------------


class TestDateWindowStrategy:
    def test_lookback_days_builds_trans_date_query(self):
        params = {"timestamp_field": "trans_date", "lookback_days": 30}
        queries = _build_date_window_queries(params)

        assert len(queries) == 1
        q = queries[0]
        assert "trans_date" in q
        assert "$gte" in q["trans_date"]
        assert "$lt" in q["trans_date"]

        start_dt = q["trans_date"]["$gte"]
        end_dt = q["trans_date"]["$lt"]
        assert isinstance(start_dt, datetime)
        assert isinstance(end_dt, datetime)
        delta = end_dt - start_dt
        assert 29 <= delta.days <= 31  # allow 1-second rounding slack

    def test_explicit_start_end_dates_build_trans_date_query(self):
        params = {
            "timestamp_field": "trans_date",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        }
        queries = _build_date_window_queries(params)

        assert len(queries) == 1
        q = queries[0]
        assert "trans_date" in q
        start_dt = q["trans_date"]["$gte"]
        end_dt = q["trans_date"]["$lt"]
        assert start_dt.year == 2024 and start_dt.month == 1 and start_dt.day == 1
        assert end_dt.year == 2024 and end_dt.month == 4 and end_dt.day == 1

    def test_default_timestamp_field_is_trans_date(self):
        # No timestamp_field in params → should default to trans_date
        params = {"lookback_days": 7}
        queries = _build_date_window_queries(params)
        assert "trans_date" in queries[0]

    def test_raises_on_missing_required_params(self):
        params = {}  # neither lookback_days nor start_date+end_date
        with pytest.raises(FraudDetectionException):
            _build_date_window_queries(params)

    def test_raises_when_only_start_date_provided(self):
        params = {"start_date": "2024-01-01"}  # missing end_date
        with pytest.raises(FraudDetectionException):
            _build_date_window_queries(params)

    def test_via_build_query_batches_from_strategy(self):
        params = {"lookback_days": 14}
        queries = build_query_batches_from_strategy("date_window", params)
        assert len(queries) == 1
        assert "trans_date" in queries[0]


# ---------------------------------------------------------------------------
# member_list strategy
# ---------------------------------------------------------------------------


class TestMemberListStrategy:
    def test_inline_source_builds_in_query(self):
        params = {
            "member_ids_source": "inline",
            "member_ids": ["M001", "M002", "M003"],
        }
        queries = _build_member_list_queries(params)

        assert len(queries) == 1
        assert queries[0] == {"member_id": {"$in": ["M001", "M002", "M003"]}}

    def test_chunking_splits_large_list_deterministically(self):
        # 25,000 IDs → chunks of 10k, 10k, 5k
        member_ids = [f"M{i:06d}" for i in range(25_000)]
        params = {
            "member_ids_source": "inline",
            "member_ids": member_ids,
        }
        queries = _build_member_list_queries(params)

        assert len(queries) == 3
        assert len(queries[0]["member_id"]["$in"]) == 10_000
        assert len(queries[1]["member_id"]["$in"]) == 10_000
        assert len(queries[2]["member_id"]["$in"]) == 5_000

        # Order is deterministic — first chunk starts at M000000
        assert queries[0]["member_id"]["$in"][0] == "M000000"
        assert queries[1]["member_id"]["$in"][0] == "M010000"
        assert queries[2]["member_id"]["$in"][0] == "M020000"

    def test_member_list_from_file_reads_correctly(self, tmp_path):
        csv_path = tmp_path / "members.csv"
        pd.DataFrame({"member_id": ["A1", "A2", "A3"]}).to_csv(csv_path, index=False)

        params = {
            "member_ids_source": "file",
            "member_ids_file": str(csv_path),
            "member_ids_column": "member_id",
        }
        ids = _resolve_member_ids(params)
        assert sorted(ids) == ["A1", "A2", "A3"]

    def test_member_list_from_fraud_csv(self, tmp_path):
        csv_path = tmp_path / "fraud.csv"
        pd.DataFrame({"member_id": ["F1", "F2"], "draw_id": [10, 20]}).to_csv(
            csv_path, index=False
        )

        params = {"member_ids_source": "fraud_csv"}
        ids = _resolve_member_ids(params, fraud_csv_path=csv_path)
        assert sorted(ids) == ["F1", "F2"]

    def test_inline_empty_list_raises(self):
        params = {"member_ids_source": "inline", "member_ids": []}
        with pytest.raises(FraudDetectionException):
            _resolve_member_ids(params)

    def test_file_source_missing_file_path_raises(self):
        params = {"member_ids_source": "file"}  # no member_ids_file
        with pytest.raises(FraudDetectionException):
            _resolve_member_ids(params)

    def test_fraud_csv_source_missing_path_raises(self):
        params = {"member_ids_source": "fraud_csv"}  # no path anywhere
        with pytest.raises(FraudDetectionException):
            _resolve_member_ids(params, fraud_csv_path=None)

    def test_unknown_source_raises(self):
        params = {"member_ids_source": "s3"}
        with pytest.raises(FraudDetectionException):
            _resolve_member_ids(params)


# ---------------------------------------------------------------------------
# full_collection strategy
# ---------------------------------------------------------------------------


class TestFullCollectionStrategy:
    def test_requires_confirm_full_pull_true(self):
        params = {"confirm_full_pull": False}
        with pytest.raises(FraudDetectionException):
            _validate_full_pull_confirmed(params)

    def test_raises_when_confirm_missing(self):
        params = {}
        with pytest.raises(FraudDetectionException):
            _validate_full_pull_confirmed(params)

    def test_returns_empty_filter_when_confirmed(self):
        params = {"confirm_full_pull": True}
        queries = build_query_batches_from_strategy("full_collection", params)
        assert queries == [{}]

    def test_via_build_query_batches_raises_without_confirm(self):
        params = {"confirm_full_pull": False}
        with pytest.raises(FraudDetectionException):
            build_query_batches_from_strategy("full_collection", params)


# ---------------------------------------------------------------------------
# Unknown strategy
# ---------------------------------------------------------------------------


class TestUnknownStrategy:
    def test_unknown_strategy_raises(self):
        with pytest.raises(FraudDetectionException):
            build_query_batches_from_strategy("watermark_sync", {})

    def test_empty_strategy_name_raises(self):
        with pytest.raises(FraudDetectionException):
            build_query_batches_from_strategy("", {})
