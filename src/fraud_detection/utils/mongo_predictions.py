from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from fraud_detection.constants.constants import ENV_MONGODB_DATABASE, ENV_MONGODB_URI
from fraud_detection.utils.mongodb import get_serving_mongo_collection


ENV_LIVE_PREDICTIONS_COLLECTION = "LIVE_PREDICTIONS_COLLECTION"
ENV_LIVE_SCORING_STATE_COLLECTION = "LIVE_SCORING_STATE_COLLECTION"


def get_live_predictions_collection():
    return get_serving_mongo_collection(
        ENV_MONGODB_URI,
        ENV_MONGODB_DATABASE,
        ENV_LIVE_PREDICTIONS_COLLECTION,
    )


def get_live_scoring_state_collection():
    return get_serving_mongo_collection(
        ENV_MONGODB_URI,
        ENV_MONGODB_DATABASE,
        ENV_LIVE_SCORING_STATE_COLLECTION,
    )


def ensure_prediction_indexes() -> None:
    predictions = get_live_predictions_collection()
    state = get_live_scoring_state_collection()
    predictions.create_index([("draw_id", 1), ("source_run_id", 1)], unique=True)
    predictions.create_index("scored_at")
    predictions.create_index("requires_review")
    predictions.create_index("flagged_members.member_id")
    state.create_index("_id", unique=True)


def upsert_draw_prediction(doc: dict[str, Any]) -> Any:
    collection = get_live_predictions_collection()
    result = collection.update_one(
        {"draw_id": int(doc["draw_id"]), "source_run_id": doc.get("source_run_id")},
        {"$set": doc},
        upsert=True,
    )
    return result.upserted_id


def read_stage1_history(*, source_run_id: str | None, since: datetime, until: datetime | None = None) -> pd.DataFrame:
    query: dict[str, Any] = {"scored_at": {"$gte": since.isoformat()}}
    if until is not None:
        query["scored_at"]["$lt"] = until.isoformat()
    if source_run_id is not None:
        query["source_run_id"] = source_run_id
    docs = list(get_live_predictions_collection().find(query, {"_id": 0, "member_scores": 1}))
    rows: list[dict[str, Any]] = []
    for doc in docs:
        for score in doc.get("member_scores", []) or []:
            rows.append(
                {
                    "member_id": str(score.get("member_id", "")).strip().upper(),
                    "draw_id": score.get("draw_id"),
                    "draw_date": score.get("draw_date"),
                    "best_partner_member_id": score.get("best_partner_member_id"),
                    "stage1_score": score.get("stage1_score", 0.0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["member_id", "draw_id", "draw_date", "best_partner_member_id", "stage1_score"])
    frame = pd.DataFrame(rows)
    frame["draw_date"] = pd.to_datetime(frame["draw_date"], errors="coerce", utc=True)
    frame["stage1_score"] = pd.to_numeric(frame["stage1_score"], errors="coerce").fillna(0.0)
    return frame
