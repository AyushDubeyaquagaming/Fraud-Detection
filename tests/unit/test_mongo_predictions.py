from __future__ import annotations

from datetime import datetime, timezone

from fraud_detection.utils import mongo_predictions


class _FakeCollection:
    def __init__(self):
        self.queries: list[dict] = []

    def find(self, query, projection):
        self.queries.append(query)
        return [{"draw_id": 1, "member_id": "A", "label": "fraud"}]


def test_read_analyst_labels_empty_draw_list_returns_no_labels(monkeypatch):
    def fail_collection_lookup():
        raise AssertionError("empty draw list should not query analyst labels collection")

    monkeypatch.setattr(mongo_predictions, "get_analyst_labels_collection", fail_collection_lookup)

    assert mongo_predictions.read_analyst_labels_for_draws([]) == []


def test_read_analyst_labels_none_preserves_all_labels_query(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(mongo_predictions, "get_analyst_labels_collection", lambda: collection)

    labels = mongo_predictions.read_analyst_labels_for_draws(None)

    assert labels == [{"draw_id": 1, "member_id": "A", "label": "fraud"}]
    assert collection.queries == [{}]


def test_read_analyst_labels_scopes_to_draw_ids(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(mongo_predictions, "get_analyst_labels_collection", lambda: collection)

    mongo_predictions.read_analyst_labels_for_draws([2, 3])

    assert collection.queries == [{"draw_id": {"$in": [2, 3]}}]


class _FakeUpdateResult:
    upserted_id = "new-label-id"


class _FakeUpsertCollection:
    def __init__(self):
        self.filter = None
        self.update = None
        self.upsert = None

    def update_one(self, filter_doc, update_doc, upsert):
        self.filter = filter_doc
        self.update = update_doc
        self.upsert = upsert
        return _FakeUpdateResult()

    def find_one(self, *_args, **_kwargs):
        return {"_id": "existing-label-id"}


def test_upsert_analyst_label_stores_draw_date_and_ccs_id_without_analyst_id(monkeypatch):
    collection = _FakeUpsertCollection()
    monkeypatch.setattr(mongo_predictions, "get_analyst_labels_collection", lambda: collection)

    label_id, created = mongo_predictions.upsert_analyst_label(
        draw_id=7297697,
        member_id=" gk00236424 ",
        label="fraud",
        analyst_id="legacy-analyst",
        model_version="partnership_v1",
        draw_date=datetime(2026, 5, 4, tzinfo=timezone.utc),
        ccs_id=" ccs015695 ",
    )

    stored = collection.update["$set"]
    assert label_id == "new-label-id"
    assert created is True
    assert collection.filter == {"draw_id": 7297697, "member_id": "GK00236424"}
    assert collection.upsert is True
    assert stored["draw_date"] == datetime(2026, 5, 4, tzinfo=timezone.utc)
    assert stored["ccs_id"] == "CCS015695"
    assert "analyst_id" not in stored
    assert collection.update["$unset"] == {"analyst_id": ""}
