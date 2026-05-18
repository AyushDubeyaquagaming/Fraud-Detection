from __future__ import annotations

from typing import Any

import pandas as pd


TIMESTAMP_CANDIDATES = [
    "createdAt.$date",
    "createdat.$date",
    "trans_date.$date",
    "updatedAt.$date",
    "ts",
    "createdAt",
    "trans_date",
    "updatedAt",
    "draw_date",
    "trans_date_min",
]


def coerce_mongo_datetime(value: Any) -> Any:
    if isinstance(value, dict) and "$date" in value:
        return value["$date"]
    return value


def normalize_event_timestamp(df: pd.DataFrame, candidates: list[str] | None = None) -> pd.Series:
    """Return the first parseable UTC timestamp across common event timestamp columns."""
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for column in candidates or TIMESTAMP_CANDIDATES:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column].map(coerce_mongo_datetime), utc=True, errors="coerce")
        ts = ts.fillna(parsed)
    return ts
