from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from prefect import flow, get_run_logger

    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False

from fraud_detection.constants.constants import (
    CONFIG_FILE_PATH,
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
    MODEL_BUNDLE_FILE,
)
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
from fraud_detection.utils.common import load_joblib, read_json, read_yaml
from fraud_detection.utils.mongo_predictions import (
    ensure_prediction_indexes,
    get_live_scoring_state_collection,
    read_stage1_history,
    upsert_draw_prediction,
)
from fraud_detection.utils.mongodb import get_serving_mongo_collection
from orchestration.notifications import notify_failure


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_scorer(config: dict) -> tuple[DrawScorer, str | None]:
    current_dir = _resolve_repo_path(config["pipeline"]["current_dir"])
    bundle = load_joblib(current_dir / MODEL_BUNDLE_FILE)
    manifest_path = current_dir / str(config.get("serving", {}).get("manifest_file", "serving_manifest.json"))
    source_run_id = None
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        source_run_id = manifest.get("run_id")
        partnership_table_path = current_dir / str(manifest.get("partnership_table_file", "partnership_table.parquet"))
    else:
        partnership_table_path = current_dir / "partnership_table.parquet"
    partnership_table = pd.read_parquet(partnership_table_path) if partnership_table_path.exists() else pd.DataFrame()
    return DrawScorer(bundle, source_run_id=source_run_id, partnership_table=partnership_table), source_run_id


def _score_new_draws(config_path: str | Path) -> dict:
    config = read_yaml(Path(config_path))
    live_cfg = config.get("live_scoring", {})
    timestamp_field = str(live_cfg.get("timestamp_field", "trans_date"))
    settle_lag_seconds = int(live_cfg.get("settle_lag_seconds", 60))
    initial_lookback_seconds = int(live_cfg.get("initial_lookback_seconds", 3600))
    max_draws_per_run = int(live_cfg.get("max_draws_per_run", 200))
    stage1_history_days = int(live_cfg.get("stage1_history_days", 7))

    ensure_prediction_indexes()
    scorer, source_run_id = _load_scorer(config)
    raw_collection = get_serving_mongo_collection(ENV_MONGODB_URI, ENV_MONGODB_DATABASE, ENV_MONGODB_COLLECTION)
    state_collection = get_live_scoring_state_collection()

    now = datetime.now(timezone.utc)
    end = now - timedelta(seconds=settle_lag_seconds)
    state = state_collection.find_one({"_id": "watermark"}) or {}
    start = state.get("trans_date_high")
    if start is None:
        start = end - timedelta(seconds=initial_lookback_seconds)
    start = pd.Timestamp(start).to_pydatetime()

    query = {timestamp_field: {"$gt": start, "$lte": end}}
    draw_ids = sorted(raw_collection.distinct("draw_id", query))[:max_draws_per_run]
    stage1_history = read_stage1_history(
        source_run_id=source_run_id,
        since=end - timedelta(days=stage1_history_days),
        until=end,
    )
    scored = 0
    failures: list[dict] = []
    for draw_id in draw_ids:
        rows = list(raw_collection.find({"draw_id": draw_id}, {"_id": 0}))
        if not rows:
            continue
        try:
            doc = scorer.score_draw(pd.DataFrame(rows), stage1_history=stage1_history).to_mongo_doc()
            doc["draw_completed_at"] = max(
                pd.to_datetime([row.get(timestamp_field) for row in rows], errors="coerce", utc=True)
            ).to_pydatetime()
            upsert_draw_prediction(doc)
            scored += 1
        except Exception as exc:
            failures.append({"draw_id": draw_id, "error": str(exc)})

    state_collection.update_one(
        {"_id": "watermark"},
        {
            "$set": {
                "trans_date_high": end,
                "last_run_at": now,
                "last_processed_draws": draw_ids,
                "last_scored_count": scored,
                "last_failure_count": len(failures),
                "source_run_id": source_run_id,
            }
        },
        upsert=True,
    )
    return {
        "status": "FINISHED",
        "draws_seen": len(draw_ids),
        "draws_scored": scored,
        "failures": failures,
        "watermark": end.isoformat(),
    }


if _PREFECT_AVAILABLE:

    @flow(name="fraud-detection-live-scoring", log_prints=True)
    def live_scoring_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:
        logger = get_run_logger()
        try:
            result = _score_new_draws(config_path)
            logger.info("Live scoring complete: %s", json.dumps(result, default=str))
            return result
        except Exception as exc:
            notify_failure(flow_name="fraud-detection-live-scoring", error=str(exc))
            raise

else:

    def live_scoring_flow(config_path: str = str(CONFIG_FILE_PATH)) -> dict:  # type: ignore[misc]
        return _score_new_draws(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run partnership live scoring once")
    parser.add_argument("--config", default=str(CONFIG_FILE_PATH))
    args = parser.parse_args()
    print(json.dumps(live_scoring_flow(config_path=args.config), indent=2, default=str))
