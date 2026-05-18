from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from fraud_detection.pipeline import batch_scoring_pipeline


class _Result:
    def __init__(self, draw_id: int):
        self.draw_id = draw_id

    def to_mongo_doc(self) -> dict:
        return {"draw_id": self.draw_id}


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
    monkeypatch.setattr(batch_scoring_pipeline, "load_joblib", lambda _path: {"use_candidate_store": True, "model_version": "partnership_v1"})
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
