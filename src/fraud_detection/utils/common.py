from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def write_json(data: Any, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Wrote JSON: %s", path)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug("Saved parquet (%d rows): %s", len(df), path)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
        logger.debug("Loaded parquet (%d rows): %s", len(df), path)
        return df
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_joblib(obj: Any, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.debug("Saved joblib: %s", path)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def load_joblib(path: Path) -> Any:
    try:
        obj = joblib.load(path)
        logger.debug("Loaded joblib: %s", path)
        return obj
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
