from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fraud_detection.utils.time_utils import normalize_event_timestamp

MATCHED = "MATCHED"
DROPPED_OUT_OF_WINDOW = "DROPPED_OUT_OF_WINDOW"
DROPPED_NOT_IN_SOURCE = "DROPPED_NOT_IN_SOURCE"
DROPPED_KEY_MISMATCH = "DROPPED_KEY_MISMATCH"
DROPPED_NO_HISTORY = "DROPPED_NO_HISTORY"


@dataclass(frozen=True)
class TimestampBounds:
    start: pd.Timestamp | None
    end: pd.Timestamp | None


def normalize_fraud_csv(fraud_csv_df: pd.DataFrame) -> pd.DataFrame:
    fraud = fraud_csv_df.copy()
    fraud.columns = [str(column).strip().lower() for column in fraud.columns]
    if not {"member_id", "draw_id"}.issubset(fraud.columns):
        raise ValueError(f"fraud CSV must contain member_id and draw_id columns; found {fraud.columns.tolist()}")
    fraud["member_id_norm"] = fraud["member_id"].astype(str).str.strip().str.upper()
    fraud["draw_id_norm"] = pd.to_numeric(fraud["draw_id"], errors="coerce").astype("Int64")
    fraud["date_parsed"] = pd.to_datetime(fraud["date"], errors="coerce", utc=True) if "date" in fraud.columns else pd.NaT
    return fraud


def normalize_available_keys(available_keys_df: pd.DataFrame) -> pd.DataFrame:
    available = available_keys_df.copy()
    if "member_id" not in available.columns or "draw_id" not in available.columns:
        raise ValueError("available_keys_df must contain member_id and draw_id columns")
    available["member_id_norm"] = available["member_id"].astype(str).str.strip().str.upper()
    available["draw_id_norm"] = pd.to_numeric(available["draw_id"], errors="coerce").astype("Int64")
    available["ts"] = normalize_event_timestamp(available)
    return available.dropna(subset=["member_id_norm", "draw_id_norm"])


def infer_timestamp_bounds(available: pd.DataFrame, ts_bounds: dict[str, Any] | TimestampBounds | None = None) -> TimestampBounds:
    if isinstance(ts_bounds, TimestampBounds):
        return ts_bounds
    if isinstance(ts_bounds, dict):
        start = ts_bounds.get("start") or ts_bounds.get("min")
        end = ts_bounds.get("end") or ts_bounds.get("max")
        return TimestampBounds(
            pd.to_datetime(start, errors="coerce", utc=True) if start is not None else None,
            pd.to_datetime(end, errors="coerce", utc=True) if end is not None else None,
        )
    valid = pd.to_datetime(available.get("ts", pd.Series(dtype=object)), errors="coerce", utc=True).dropna()
    return TimestampBounds(valid.min() if not valid.empty else None, valid.max() if not valid.empty else None)


def classify_fraud_csv(
    fraud_csv_df: pd.DataFrame,
    available_keys_df: pd.DataFrame,
    ts_bounds: dict[str, Any] | TimestampBounds | None = None,
) -> pd.DataFrame:
    """Classify fraud CSV rows against a source-agnostic `(member_id, draw_id, ts)` availability table."""
    fraud = normalize_fraud_csv(fraud_csv_df)
    available = normalize_available_keys(available_keys_df)
    bounds = infer_timestamp_bounds(available, ts_bounds)
    members_in_source = set(available["member_id_norm"].dropna())
    event_keys = set(zip(available["member_id_norm"], available["draw_id_norm"].astype("Int64")))
    history_counts = _history_counts(available, fraud)

    rows: list[dict[str, Any]] = []
    for fraud_row in fraud.to_dict("records"):
        member_id = str(fraud_row["member_id_norm"])
        draw_id = fraud_row["draw_id_norm"]
        fraud_date = fraud_row.get("date_parsed")
        found_member = member_id in members_in_source
        matched_key = (member_id, draw_id) in event_keys
        rows_before = int(history_counts.get(member_id, 0))
        status = _status(
            found_member=found_member,
            matched_key=matched_key,
            rows_before=rows_before,
            fraud_date=fraud_date,
            bounds=bounds,
        )
        rows.append(
            {
                "fraud_csv_member_id": member_id,
                "fraud_csv_draw_id": None if pd.isna(draw_id) else int(draw_id),
                "fraud_csv_date": None if pd.isna(fraud_date) else pd.Timestamp(fraud_date).isoformat(),
                "found_in_source": bool(found_member),
                "matched_event_key": bool(matched_key),
                "rows_before_cutoff": rows_before,
                "source_window_start": None if bounds.start is None or pd.isna(bounds.start) else bounds.start.isoformat(),
                "source_window_end": None if bounds.end is None or pd.isna(bounds.end) else bounds.end.isoformat(),
                "final_status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize_verdicts(verdict_df: pd.DataFrame) -> dict[str, Any]:
    counts = verdict_df.get("final_status", pd.Series(dtype=object)).value_counts().to_dict()
    normalized_counts = {str(key): int(value) for key, value in counts.items()}
    return {
        "fraud_csv_rows": int(len(verdict_df)),
        "fraud_csv_unique_members": int(verdict_df.get("fraud_csv_member_id", pd.Series(dtype=object)).nunique()),
        "event_level_status_counts": normalized_counts,
        "event_status_counts": normalized_counts,
    }


def _history_counts(available: pd.DataFrame, fraud: pd.DataFrame) -> dict[str, int]:
    available_valid = available.dropna(subset=["ts"]).copy()
    fraud_dates = fraud.dropna(subset=["date_parsed"]).groupby("member_id_norm")["date_parsed"].min()
    counts: dict[str, int] = {}
    if available_valid.empty or fraud_dates.empty:
        return counts
    for member_id, cutoff in fraud_dates.items():
        member_rows = available_valid.loc[available_valid["member_id_norm"].eq(member_id)]
        counts[str(member_id)] = int((member_rows["ts"] < cutoff).sum())
    return counts


def _status(
    *,
    found_member: bool,
    matched_key: bool,
    rows_before: int,
    fraud_date: Any,
    bounds: TimestampBounds,
) -> str:
    if not found_member:
        return DROPPED_NOT_IN_SOURCE
    if not matched_key:
        if pd.notna(fraud_date) and bounds.start is not None and bounds.end is not None:
            fraud_ts = pd.Timestamp(fraud_date)
            if fraud_ts < bounds.start or fraud_ts > bounds.end:
                return DROPPED_OUT_OF_WINDOW
        return DROPPED_KEY_MISMATCH
    if pd.notna(fraud_date) and rows_before <= 0:
        return DROPPED_NO_HISTORY
    return MATCHED
