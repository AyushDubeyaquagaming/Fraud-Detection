from __future__ import annotations

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
