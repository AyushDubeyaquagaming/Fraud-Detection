from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fraud_detection.extraction import ccs_profit_aggregator
from fraud_detection.extraction.ccs_profit_aggregator import aggregate_profit_documents, build_ccs_daily_profit, write_partitioned_profit


def test_synthetic_grouped_output_is_correct():
    result = aggregate_profit_documents(
        [
            {"ccs_id": "c1", "member_id": "a", "draw_id": 1, "trans_date": "2026-05-07T01:00:00Z", "total_bet_amount": 100.0, "win_points": 150.0},
            {"ccs_id": "C1", "member_id": "A", "draw_id": 2, "trans_date": "2026-05-07T02:00:00Z", "total_bet_amount": 50.0, "win_points": 0.0},
            {"ccs_id": "C1", "member_id": "B", "draw_id": 3, "trans_date": "2026-05-07T03:00:00Z", "total_bet_amount": 10.0, "win_points": 20.0},
        ]
    )

    row = result.loc[result["member_id"].eq("A")].iloc[0]
    assert row["daily_bet_amount"] == 150.0
    assert row["daily_win_points"] == 150.0
    assert row["daily_profit"] == 0.0
    assert row["draw_count"] == 2


def test_aggregation_skips_missing_identifiers():
    result = aggregate_profit_documents(
        [
            {"ccs_id": None, "member_id": "A", "trans_date": "2026-05-07T01:00:00Z", "total_bet_amount": 100.0, "win_points": 150.0},
            {"ccs_id": "C1", "member_id": "nan", "trans_date": "2026-05-07T02:00:00Z", "total_bet_amount": 100.0, "win_points": 150.0},
            {"ccs_id": "C1", "member_id": "A", "trans_date": "2026-05-07T03:00:00Z", "total_bet_amount": 50.0, "win_points": 90.0},
        ]
    )

    assert result[["ccs_id", "member_id"]].to_dict("records") == [{"ccs_id": "C1", "member_id": "A"}]
    assert result["daily_profit"].iloc[0] == 40.0


def test_mongo_profit_rows_skip_missing_identifiers():
    result = ccs_profit_aggregator._normalize_profit_frame(
        [
            {
                "_id": {"ccs_id": None, "member_id": "A", "profit_date": "2026-05-07"},
                "daily_bet_amount": 100.0,
                "daily_win_points": 200.0,
                "daily_profit": 100.0,
                "draw_count": 1,
            },
            {
                "_id": {"ccs_id": "C1", "member_id": "A", "profit_date": "2026-05-07"},
                "daily_bet_amount": 50.0,
                "daily_win_points": 90.0,
                "daily_profit": 40.0,
                "draw_count": 1,
            },
        ]
    )

    assert result[["ccs_id", "member_id"]].to_dict("records") == [{"ccs_id": "C1", "member_id": "A"}]


def test_partition_write_is_resume_safe(tmp_path):
    frame = pd.DataFrame(
        {
            "ccs_id": ["C1"],
            "member_id": ["A"],
            "profit_date": [pd.Timestamp("2026-05-07").date()],
            "daily_profit": [1.0],
            "daily_bet_amount": [2.0],
            "daily_win_points": [3.0],
            "draw_count": [1],
        }
    )

    write_partitioned_profit(frame, tmp_path)
    target = tmp_path / "year=2026" / "month=05" / "ccs_daily_profit.parquet"
    write_partitioned_profit(frame.assign(daily_profit=[999.0]), tmp_path)

    assert pd.read_parquet(target)["daily_profit"].iloc[0] == 999.0


class _FakeCollection:
    def __init__(self):
        self.matched_days: list[date] = []

    def aggregate(self, pipeline, allowDiskUse=False):
        assert allowDiskUse is True
        match = pipeline[0]["$match"]["trans_date"]
        day = match["$gte"].date()
        self.matched_days.append(day)
        return [
            {
                "_id": {"ccs_id": "C1", "member_id": "A", "profit_date": day.isoformat()},
                "daily_bet_amount": 100.0,
                "daily_win_points": 180.0,
                "daily_profit": 80.0,
                "draw_count": 2,
            }
        ]


class _ReplacingCollection:
    def __init__(self, rows_by_day):
        self.rows_by_day = rows_by_day
        self.matched_days: list[date] = []

    def aggregate(self, pipeline, allowDiskUse=False):
        assert allowDiskUse is True
        match = pipeline[0]["$match"]["trans_date"]
        day = match["$gte"].date()
        self.matched_days.append(day)
        return self.rows_by_day.get(day, [])


def test_build_ccs_daily_profit_writes_each_day(monkeypatch, tmp_path):
    collection = _FakeCollection()
    seen_env_vars = []

    def fake_get_collection(*args):
        seen_env_vars.append(args)
        return collection

    monkeypatch.setattr(ccs_profit_aggregator, "get_serving_mongo_collection", fake_get_collection)

    summary = build_ccs_daily_profit(
        date(2026, 5, 6),
        date(2026, 5, 7),
        str(tmp_path),
        uri_env_var="CUSTOM_URI",
        database_env_var="CUSTOM_DATABASE",
        collection_env_var="CUSTOM_COLLECTION",
    )

    target = tmp_path / "year=2026" / "month=05" / "ccs_daily_profit.parquet"
    result = pd.read_parquet(target)
    assert seen_env_vars == [("CUSTOM_URI", "CUSTOM_DATABASE", "CUSTOM_COLLECTION")]
    assert collection.matched_days == [date(2026, 5, 6), date(2026, 5, 7)]
    assert sorted(result["profit_date"].astype(str).tolist()) == ["2026-05-06", "2026-05-07"]
    assert summary.status == "FINISHED"
    assert summary.days_succeeded == 2
    assert summary.rows_written == 2


def test_build_ccs_daily_profit_skips_existing_days_without_force(monkeypatch, tmp_path):
    existing = pd.DataFrame(
        {
            "ccs_id": ["C1"],
            "member_id": ["A"],
            "profit_date": [date(2026, 5, 6)],
            "daily_profit": [80.0],
            "daily_bet_amount": [100.0],
            "daily_win_points": [180.0],
            "draw_count": [2],
        }
    )
    write_partitioned_profit(existing, tmp_path)
    collection = _FakeCollection()
    monkeypatch.setattr(ccs_profit_aggregator, "get_serving_mongo_collection", lambda *args: collection)

    summary = build_ccs_daily_profit(date(2026, 5, 6), date(2026, 5, 7), str(tmp_path))

    assert collection.matched_days == [date(2026, 5, 7)]
    assert summary.skipped_days == 1
    assert summary.days_succeeded == 1


def test_force_rebuild_replaces_stale_rows_for_rebuilt_day(monkeypatch, tmp_path):
    existing = pd.DataFrame(
        {
            "ccs_id": ["C1", "C1", "C1"],
            "member_id": ["A", "B", "KEEP"],
            "profit_date": [date(2026, 5, 6), date(2026, 5, 6), date(2026, 5, 7)],
            "daily_profit": [10.0, 20.0, 30.0],
            "daily_bet_amount": [100.0, 100.0, 100.0],
            "daily_win_points": [110.0, 120.0, 130.0],
            "draw_count": [1, 1, 1],
        }
    )
    write_partitioned_profit(existing, tmp_path)
    collection = _ReplacingCollection(
        {
            date(2026, 5, 6): [
                {
                    "_id": {"ccs_id": "C1", "member_id": "A", "profit_date": "2026-05-06"},
                    "daily_bet_amount": 200.0,
                    "daily_win_points": 260.0,
                    "daily_profit": 60.0,
                    "draw_count": 2,
                }
            ]
        }
    )
    monkeypatch.setattr(ccs_profit_aggregator, "get_serving_mongo_collection", lambda *args: collection)

    summary = build_ccs_daily_profit(date(2026, 5, 6), date(2026, 5, 6), str(tmp_path), force=True)

    result = pd.read_parquet(tmp_path / "year=2026" / "month=05" / "ccs_daily_profit.parquet")
    day6 = result.loc[pd.to_datetime(result["profit_date"]).dt.date.eq(date(2026, 5, 6))]
    assert summary.days_succeeded == 1
    assert day6[["member_id", "daily_profit"]].to_dict("records") == [{"member_id": "A", "daily_profit": 60.0}]
    assert "KEEP" in set(result["member_id"])


def test_force_rebuild_removes_stale_day_when_mongo_returns_no_rows(monkeypatch, tmp_path):
    existing = pd.DataFrame(
        {
            "ccs_id": ["C1", "C1"],
            "member_id": ["STALE", "KEEP"],
            "profit_date": [date(2026, 5, 6), date(2026, 5, 7)],
            "daily_profit": [10.0, 30.0],
            "daily_bet_amount": [100.0, 100.0],
            "daily_win_points": [110.0, 130.0],
            "draw_count": [1, 1],
        }
    )
    write_partitioned_profit(existing, tmp_path)
    collection = _ReplacingCollection({date(2026, 5, 6): []})
    monkeypatch.setattr(ccs_profit_aggregator, "get_serving_mongo_collection", lambda *args: collection)

    summary = build_ccs_daily_profit(date(2026, 5, 6), date(2026, 5, 6), str(tmp_path), force=True)

    result = pd.read_parquet(tmp_path / "year=2026" / "month=05" / "ccs_daily_profit.parquet")
    assert summary.days_succeeded == 1
    assert summary.rows_written == 0
    assert set(result["member_id"]) == {"KEEP"}
    assert not pd.to_datetime(result["profit_date"]).dt.date.eq(date(2026, 5, 6)).any()


def test_build_ccs_daily_profit_rejects_invalid_window(tmp_path):
    with pytest.raises(ValueError, match="must be on or before"):
        build_ccs_daily_profit(date(2026, 5, 8), date(2026, 5, 7), str(tmp_path))
