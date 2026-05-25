from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from fraud_detection.pipeline import batch_scoring_pipeline


class _Result:
    def __init__(self, draw_id: int, doc: dict | None = None):
        self.draw_id = draw_id
        self.doc = doc

    def to_mongo_doc(self) -> dict:
        return self.doc or {"draw_id": self.draw_id}


def test_parquet_doc_writer_handles_empty_then_populated_nested_lists(tmp_path: Path):
    output_path = tmp_path / "live_predictions_backfill.parquet"
    writer = batch_scoring_pipeline._ParquetDocWriter(output_path)
    writer.write(
        [
            {
                "draw_id": 1,
                "scored_at": "2026-05-25T00:00:00+00:00",
                "model_version": "partnership_v1",
                "source_run_id": "run_a",
                "n_members_in_draw": 0,
                "candidate_members": [],
                "partnerships": [],
                "flagged_members": [],
                "member_scores": [],
                "max_stage1_score": 0.0,
                "max_stage2_score": 0.0,
                "requires_review": False,
                "response_details": [],
            }
        ]
    )
    writer.write(
        [
            {
                "draw_id": 2,
                "scored_at": "2026-05-25T00:01:00+00:00",
                "model_version": "partnership_v1",
                "source_run_id": "run_a",
                "n_members_in_draw": 2,
                "candidate_members": ["A", "B"],
                "partnerships": [
                    {
                        "member_ids": ["A", "B"],
                        "stage1_score_max": 1.0,
                        "stage1_score_mean": 1.0,
                        "union_coverage": 1.0,
                        "jaccard": 0.0,
                        "per_position_ratio": 1.0,
                        "total_stake_ratio": 1.0,
                        "combined_bet_cv": 0.0,
                        "pair_net": 100.0,
                        "rule_confidence": 1.0,
                        "is_section_a": False,
                        "is_section_b": True,
                    }
                ],
                "flagged_members": [
                    {
                        "member_id": "A",
                        "stage1_score_in_draw": 1.0,
                        "stage2_score": 0.0,
                        "best_partner_member_id": "B",
                        "bet_amount": 1000.0,
                        "win_amount": 0.0,
                        "high_amount_flag": False,
                        "high_amount_reason": None,
                    }
                ],
                "member_scores": [
                    {
                        "member_id": "A",
                        "draw_id": 2,
                        "draw_date": "2026-05-25T00:00:00+00:00",
                        "best_partner_member_id": "B",
                        "stage1_score": 1.0,
                        "stage2_score": 0.0,
                    }
                ],
                "max_stage1_score": 1.0,
                "max_stage2_score": 0.0,
                "requires_review": True,
                "response_details": ["no_member_history"],
            }
        ]
    )
    writer.close()

    table = pq.read_table(output_path)
    assert table.num_rows == 2
    assert table.schema.field("candidate_members").type.value_type == batch_scoring_pipeline.pa.string()


def test_candidate_store_batch_scoring_defaults_to_weekly_mongodb_source(monkeypatch, tmp_path: Path):
    current_dir = tmp_path / "current"
    current_dir.mkdir(parents=True)
    (current_dir / "model_bundle.joblib").write_text("bundle")
    config_path = tmp_path / "batch_scoring.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"current_dir": str(current_dir)},
                "data_ingestion": {
                    "source": "parquet",
                    "parquet_path": str(tmp_path / "unused.parquet"),
                    "mongodb": {
                        "uri_env_var": "MONGODB_URI",
                        "database_env_var": "MONGODB_DATABASE",
                        "collection_env_var": "MONGODB_COLLECTION_ROULETTE_REPORT",
                    },
                },
                "batch_scoring": {
                    "window": {"timestamp_field": "trans_date", "lookback_days": 7},
                },
                "partnership": {
                    "use_candidate_store": True,
                    "candidate_store_path": str(tmp_path / "candidate_draws"),
                    "candidate_window": {"start_date": "2026-02-25", "end_date": "2026-05-05"},
                },
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    def fake_iter_mongo_draw_groups(*, mongo_config, window, now=None):
        calls["mongo_config"] = mongo_config
        calls["window"] = window
        yield pd.DataFrame(
            [
                {
                    "draw_id": 101,
                    "member_id": "A",
                    "ccs_id": "C1",
                    "total_bet_amount": 1000.0,
                    "win_points": 1200.0,
                    "bets": [],
                    "trans_date": pd.Timestamp("2026-05-12T00:00:00Z"),
                }
            ]
        )

    class FakeScorer:
        def __init__(self, *_args, **_kwargs):
            pass

        def score_draw(self, draw_rows: pd.DataFrame):
            return _Result(int(draw_rows["draw_id"].iloc[0]))

    monkeypatch.setattr(batch_scoring_pipeline, "_iter_mongo_draw_groups", fake_iter_mongo_draw_groups)
    monkeypatch.setattr(
        batch_scoring_pipeline,
        "_iter_candidate_store_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("candidate store path should not be used")),
    )
    monkeypatch.setattr(
        batch_scoring_pipeline,
        "load_joblib",
        lambda _path: {"use_candidate_store": True, "model_version": "partnership_v1"},
    )
    monkeypatch.setattr(batch_scoring_pipeline, "DrawScorer", FakeScorer)

    output_dir = batch_scoring_pipeline.BatchScoringPipeline(config_path=config_path).run()

    report = json.loads((output_dir / "batch_scoring_report.json").read_text(encoding="utf-8"))
    assert report["draws_scored"] == 1
    assert report["source"] == "mongodb"
    assert report["window"]["lookback_days"] == 7
    assert calls["mongo_config"]["collection_env_var"] == "MONGODB_COLLECTION_ROULETTE_REPORT"


def test_iter_mongo_draw_groups_uses_lookback_window_and_groups_by_draw(monkeypatch):
    class FakeCursor:
        def __init__(self, docs):
            self.docs = docs
            self.sort_fields = None

        def sort(self, fields):
            self.sort_fields = fields
            return self

        def batch_size(self, _size):
            return self

        def __iter__(self):
            return iter(self.docs)

    class FakeCollection:
        def __init__(self):
            self.queries = []
            self.sort_fields = []
            self.docs = [
                {"draw_id": 1, "member_id": "A", "trans_date": datetime(2026, 5, 10, tzinfo=timezone.utc)},
                {"draw_id": 1, "member_id": "B", "trans_date": datetime(2026, 5, 10, tzinfo=timezone.utc)},
                {"draw_id": 2, "member_id": "C", "trans_date": datetime(2026, 5, 11, tzinfo=timezone.utc)},
            ]

        def find(self, query, projection):
            self.queries.append(query)
            assert projection["draw_id"] == 1
            lower = query["trans_date"]["$gte"]
            upper = query["trans_date"]["$lt"]
            docs = [doc for doc in self.docs if lower <= doc["trans_date"] < upper]
            cursor = FakeCursor(docs)
            original_sort = cursor.sort

            def record_sort(fields):
                self.sort_fields.append(fields)
                return original_sort(fields)

            cursor.sort = record_sort
            return cursor

    fake_collection = FakeCollection()
    monkeypatch.setattr(batch_scoring_pipeline, "get_serving_mongo_collection", lambda *args: fake_collection)

    groups = list(
        batch_scoring_pipeline._iter_mongo_draw_groups(
            mongo_config={
                "uri_env_var": "MONGODB_URI",
                "database_env_var": "MONGODB_DATABASE",
                "collection_env_var": "MONGODB_COLLECTION_ROULETTE_REPORT",
            },
            window={"timestamp_field": "trans_date", "lookback_days": 7, "chunk_days": 1},
            now=datetime(2026, 5, 12, tzinfo=timezone.utc),
        )
    )

    assert len(groups) == 2
    assert groups[0]["draw_id"].tolist() == [1, 1]
    assert groups[1]["draw_id"].tolist() == [2]
    assert fake_collection.queries[0]["trans_date"]["$gte"].isoformat() == "2026-05-05T00:00:00+00:00"
    assert fake_collection.queries[-1]["trans_date"]["$lt"].isoformat() == "2026-05-12T00:00:00+00:00"
    assert fake_collection.sort_fields[0] == [("trans_date", 1), ("draw_id", 1)]


def test_batch_scoring_syncs_native_feedback_via_api(monkeypatch, tmp_path: Path):
    current_dir = tmp_path / "current"
    current_dir.mkdir(parents=True)
    (current_dir / "model_bundle.joblib").write_text("bundle")
    config_path = tmp_path / "batch_scoring.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"current_dir": str(current_dir)},
                "data_ingestion": {
                    "source": "parquet",
                    "parquet_path": str(tmp_path / "unused.parquet"),
                    "mongodb": {
                        "uri_env_var": "MONGODB_URI",
                        "database_env_var": "MONGODB_DATABASE",
                        "collection_env_var": "MONGODB_COLLECTION_ROULETTE_REPORT",
                    },
                },
                "batch_scoring": {
                    "source": "mongodb",
                    "window": {"timestamp_field": "trans_date", "lookback_days": 7},
                },
                "native_feedback": {
                    "enabled": True,
                    "api_base_url": "https://backend.example",
                    "suspicious_bulk_path": "/gk-users/suspicious/bulk",
                },
                "partnership": {"use_candidate_store": True},
            }
        ),
        encoding="utf-8",
    )

    def fake_iter_mongo_draw_groups(*, mongo_config, window, now=None):
        yield pd.DataFrame(
            [
                {
                    "draw_id": 101,
                    "member_id": "A",
                    "ccs_id": "C1",
                    "total_bet_amount": 1000.0,
                    "win_points": 1200.0,
                    "bets": [],
                    "trans_date": pd.Timestamp("2026-05-12T00:00:00Z"),
                }
            ]
        )

    class FakeScorer:
        def __init__(self, *_args, **_kwargs):
            pass

        def score_draw(self, draw_rows: pd.DataFrame):
            return _Result(
                int(draw_rows["draw_id"].iloc[0]),
                {
                    "draw_id": 101,
                    "requires_review": True,
                    "flagged_members": [{"member_id": "A"}],
                    "partnerships": [],
                },
            )

    sync_calls = {}

    def fake_sync(events, config):
        sync_calls["events"] = events
        sync_calls["config"] = config
        return {"enabled": True, "events": len(events), "payloads": 1, "status": "sent"}

    monkeypatch.setattr(batch_scoring_pipeline, "_iter_mongo_draw_groups", fake_iter_mongo_draw_groups)
    monkeypatch.setattr(
        batch_scoring_pipeline,
        "load_joblib",
        lambda _path: {"use_candidate_store": True, "model_version": "partnership_v1"},
    )
    monkeypatch.setattr(batch_scoring_pipeline, "DrawScorer", FakeScorer)
    monkeypatch.setattr(batch_scoring_pipeline, "sync_native_suspected_feedback", fake_sync)

    output_dir = batch_scoring_pipeline.BatchScoringPipeline(config_path=config_path).run()

    report = json.loads((output_dir / "batch_scoring_report.json").read_text(encoding="utf-8"))
    assert report["native_feedback"] == {"enabled": True, "events": 1, "payloads": 1, "status": "sent"}
    assert sync_calls["config"]["api_base_url"] == "https://backend.example"
    assert [(event.member_id, event.ccs_id, event.alert_date.isoformat()) for event in sync_calls["events"]] == [
        ("A", "C1", "2026-05-12")
    ]


def test_batch_scoring_keeps_report_when_native_feedback_sync_fails(monkeypatch, tmp_path: Path):
    current_dir = tmp_path / "current"
    current_dir.mkdir(parents=True)
    (current_dir / "model_bundle.joblib").write_text("bundle")
    config_path = tmp_path / "batch_scoring.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"current_dir": str(current_dir)},
                "data_ingestion": {
                    "source": "parquet",
                    "parquet_path": str(tmp_path / "unused.parquet"),
                    "mongodb": {
                        "uri_env_var": "MONGODB_URI",
                        "database_env_var": "MONGODB_DATABASE",
                        "collection_env_var": "MONGODB_COLLECTION_ROULETTE_REPORT",
                    },
                },
                "batch_scoring": {
                    "source": "mongodb",
                    "window": {"timestamp_field": "trans_date", "lookback_days": 7},
                },
                "native_feedback": {
                    "enabled": True,
                    "api_base_url": "https://backend.example",
                    "suspicious_bulk_path": "/gk-users/suspicious/bulk",
                },
                "partnership": {"use_candidate_store": True},
            }
        ),
        encoding="utf-8",
    )

    def fake_iter_mongo_draw_groups(*, mongo_config, window, now=None):
        yield pd.DataFrame(
            [
                {
                    "draw_id": 101,
                    "member_id": "A",
                    "ccs_id": "C1",
                    "total_bet_amount": 1000.0,
                    "win_points": 1200.0,
                    "bets": [],
                    "trans_date": pd.Timestamp("2026-05-12T00:00:00Z"),
                }
            ]
        )

    class FakeScorer:
        def __init__(self, *_args, **_kwargs):
            pass

        def score_draw(self, draw_rows: pd.DataFrame):
            return _Result(
                int(draw_rows["draw_id"].iloc[0]),
                {
                    "draw_id": 101,
                    "requires_review": True,
                    "flagged_members": [{"member_id": "A"}],
                    "partnerships": [],
                },
            )

    monkeypatch.setattr(batch_scoring_pipeline, "_iter_mongo_draw_groups", fake_iter_mongo_draw_groups)
    monkeypatch.setattr(
        batch_scoring_pipeline,
        "load_joblib",
        lambda _path: {"use_candidate_store": True, "model_version": "partnership_v1"},
    )
    monkeypatch.setattr(batch_scoring_pipeline, "DrawScorer", FakeScorer)
    monkeypatch.setattr(
        batch_scoring_pipeline,
        "sync_native_suspected_feedback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend down")),
    )

    output_dir = batch_scoring_pipeline.BatchScoringPipeline(config_path=config_path).run()

    report = json.loads((output_dir / "batch_scoring_report.json").read_text(encoding="utf-8"))
    assert report["draws_scored"] == 1
    assert report["native_feedback"]["status"] == "failed"
    assert "backend down" in report["native_feedback"]["error"]
