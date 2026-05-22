from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from fraud_detection.constants.constants import ENV_MONGODB_DATABASE, ENV_MONGODB_URI
from fraud_detection.logger import get_logger
from fraud_detection.utils.mongodb import get_serving_mongo_collection

logger = get_logger(__name__)

ENV_USERS_COLLECTION = "MONGODB_COLLECTION_USERS"
ENV_GK_BACKEND_API_BASE_URL = "GK_BACKEND_API_BASE_URL"


@dataclass(frozen=True)
class NativeFeedbackEvent:
    member_id: str
    ccs_id: str | None
    alert_date: date


def _member_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _ccs_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _as_utc_datetime(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _as_utc_iso_z(value: date) -> str:
    return _as_utc_datetime(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set)):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"true", "yes", "1", "fraud", "confirmed"}:
        return True
    if raw in {"false", "no", "0", "not_fraud", "not fraud", "rejected"}:
        return False
    return None


def _field_bool(doc: dict[str, Any], *field_names: str) -> bool | None:
    for field in field_names:
        if field in doc:
            return _coerce_bool(doc.get(field))
        upper = field.upper()
        if upper in doc:
            return _coerce_bool(doc.get(upper))
    return None


def _first_present(doc: dict[str, Any], *field_names: str) -> Any:
    for field in field_names:
        if field in doc and doc.get(field) is not None:
            return doc.get(field)
    return None


def _candidate_member_map(row: dict[str, Any] | pd.Series) -> dict[str, str | None]:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    raw_members = source.get("member_ids", [])
    raw_ccs_values = source.get("ccs_ids", [])
    members = [_member_key(value) for value in ([] if raw_members is None else list(raw_members))]
    ccs_values = [] if raw_ccs_values is None else list(raw_ccs_values)
    mapping: dict[str, str | None] = {}
    for idx, member_id in enumerate(members):
        if not member_id:
            continue
        ccs_value = ccs_values[idx] if idx < len(ccs_values) else None
        ccs_id = _ccs_key(ccs_value) if ccs_value is not None and str(ccs_value).strip() else None
        mapping[member_id] = ccs_id
    return mapping


def _raw_member_map(rows: pd.DataFrame) -> dict[str, str | None]:
    if rows.empty:
        return {}
    members = rows.get("member_id", pd.Series(dtype=object)).map(_member_key)
    ccs_values = rows.get("ccs_id", pd.Series(index=rows.index, dtype=object))
    mapping: dict[str, str | None] = {}
    for member_id, ccs_value in zip(members, ccs_values):
        if not member_id:
            continue
        ccs_id = _ccs_key(ccs_value) if ccs_value is not None and str(ccs_value).strip() else None
        mapping[member_id] = ccs_id
    return mapping


def _source_alert_date(source: dict[str, Any] | pd.Series | pd.DataFrame) -> date | None:
    if isinstance(source, pd.DataFrame):
        for column in ("trans_date", "createdAt", "updatedAt"):
            if column in source.columns:
                parsed = pd.to_datetime(source[column], errors="coerce", utc=True).dropna()
                if not parsed.empty:
                    return parsed.min().date()
        return None
    row = source.to_dict() if isinstance(source, pd.Series) else dict(source)
    for key in ("trans_date_min", "draw_date", "trans_date", "createdAt"):
        value = _coerce_date(row.get(key))
        if value is not None:
            return value
    return None


def extract_native_feedback_events(
    result_doc: dict[str, Any],
    source: dict[str, Any] | pd.Series | pd.DataFrame,
) -> list[NativeFeedbackEvent]:
    """Extract member/CCS/date events from a scored draw document and its source rows."""
    if not result_doc.get("requires_review"):
        return []
    if isinstance(source, pd.DataFrame):
        member_to_ccs = _raw_member_map(source)
    else:
        member_to_ccs = _candidate_member_map(source)
    alert_date = _source_alert_date(source)
    if alert_date is None:
        return []

    alert_members = {_member_key(item.get("member_id")) for item in result_doc.get("flagged_members", []) or []}
    for partnership in result_doc.get("partnerships", []) or []:
        alert_members.update(_member_key(member) for member in partnership.get("member_ids", []) or [])
    alert_members.discard("")

    events = []
    for member_id in sorted(alert_members):
        events.append(
            NativeFeedbackEvent(
                member_id=member_id,
                ccs_id=member_to_ccs.get(member_id),
                alert_date=alert_date,
            )
        )
    return events


def group_member_alert_periods(events: list[NativeFeedbackEvent]) -> dict[str, list[tuple[date, date]]]:
    dates_by_member: dict[str, set[date]] = defaultdict(set)
    for event in events:
        if event.member_id:
            dates_by_member[event.member_id].add(event.alert_date)

    grouped: dict[str, list[tuple[date, date]]] = {}
    for member_id, dates in dates_by_member.items():
        sorted_dates = sorted(dates)
        if not sorted_dates:
            continue
        periods: list[tuple[date, date]] = []
        start = previous = sorted_dates[0]
        for current in sorted_dates[1:]:
            if current <= previous + timedelta(days=1):
                previous = current
                continue
            periods.append((start, previous))
            start = previous = current
        periods.append((start, previous))
        grouped[member_id] = periods
    return grouped


def _collection_from_cfg(config: dict[str, Any], key: str, default_env_var: str):
    return get_serving_mongo_collection(
        str(config.get("uri_env_var", ENV_MONGODB_URI)),
        str(config.get("database_env_var", ENV_MONGODB_DATABASE)),
        str(config.get(key, default_env_var)),
    )


def build_suspicious_bulk_payload(events: list[NativeFeedbackEvent]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for member_id, periods in group_member_alert_periods(events).items():
        for start, end in periods:
            from_date = _as_utc_iso_z(start)
            to_date = _as_utc_iso_z(end)
            key = (member_id, from_date, to_date)
            if key in seen:
                continue
            seen.add(key)
            payloads.append(
                {
                    "memberId": member_id,
                    "fromDate": from_date,
                    "toDate": to_date,
                }
            )
    return payloads


def sync_native_suspected_feedback(
    events: list[NativeFeedbackEvent],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if not cfg.get("enabled", False):
        return {"enabled": False, "events": int(len(events))}
    if not events:
        return {"enabled": True, "events": 0, "payloads": 0, "status": "skipped_empty"}

    payload = build_suspicious_bulk_payload(events)
    if not payload:
        return {"enabled": True, "events": int(len(events)), "payloads": 0, "status": "skipped_empty_payload"}

    base_url = str(
        cfg.get("api_base_url") or os.getenv(str(cfg.get("api_base_url_env_var", ENV_GK_BACKEND_API_BASE_URL))) or ""
    ).strip()
    if not base_url:
        logger.warning(
            "native_feedback is enabled but backend API base URL is missing. "
            "Skipping suspicious bulk API sync; batch scoring will continue. "
            "Set %s or native_feedback.api_base_url to enable it.",
            cfg.get("api_base_url_env_var", ENV_GK_BACKEND_API_BASE_URL),
        )
        return {
            "enabled": True,
            "events": int(len(events)),
            "members": int(len({event.member_id for event in events})),
            "payloads": int(len(payload)),
            "status": "skipped_missing_api_base_url",
        }
    path = str(cfg.get("suspicious_bulk_path", "/gk-users/suspicious/bulk"))
    endpoint = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    timeout = float(cfg.get("request_timeout_seconds", 30))
    headers = {"Content-Type": "application/json"}

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if response.status_code < 200 or response.status_code >= 300:
        body = response.text[:500]
        raise RuntimeError(f"Suspicious bulk API failed with status {response.status_code}: {body}")
    return {
        "enabled": True,
        "events": int(len(events)),
        "members": int(len({event.member_id for event in events})),
        "payloads": int(len(payload)),
        "endpoint": endpoint,
        "status_code": int(response.status_code),
        "status": "sent",
    }


def _find_docs_by_members(collection, member_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not member_ids:
        return {}
    cursor = collection.find(
        {
            "$or": [
                {"member_id": {"$in": sorted(member_ids)}},
                {"memberId": {"$in": sorted(member_ids)}},
                {"gk_id": {"$in": sorted(member_ids)}},
                {"username": {"$in": sorted(member_ids)}},
            ]
        },
        {"_id": 0},
    )
    docs: dict[str, dict[str, Any]] = {}
    for doc in cursor:
        for key in ("member_id", "memberId", "gk_id", "username"):
            member_id = _member_key(doc.get(key))
            if member_id in member_ids:
                docs[member_id] = doc
                break
    return docs


def _suspected_periods(doc: dict[str, Any], field_name: str, member_id: str) -> list[tuple[date, date]]:
    value = doc.get(field_name)
    if value is None and field_name != "is_suspected_by_ml":
        value = _first_present(doc, "is_suspected_by_ml", "suspected_by_ml", "ml_suspected")
    suspected_flag = _coerce_bool(value)
    if suspected_flag is True:
        return [(date.min, date.max)]
    if suspected_flag is False:
        return []
    if not isinstance(value, list):
        return []
    periods = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payload_member = _member_key(item.get("memberId") or item.get("member_id") or member_id)
        if payload_member and payload_member != member_id:
            continue
        ml_flag = _coerce_bool(item.get("is_suspected_by_ml"))
        if ml_flag is False:
            continue
        start = _coerce_date(_first_present(item, "fromDate", "from_date"))
        end = _coerce_date(_first_present(item, "toDate", "to_date", "fromDate", "from_date"))
        if start is not None and end is not None:
            periods.append((start, end))
    return periods


def _date_in_periods(value: date, periods: list[tuple[date, date]]) -> bool:
    return any(start <= value <= end for start, end in periods)


def read_native_feedback_labels_for_candidate_rows(
    candidate_rows: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or {}
    if not cfg.get("enabled", False) or candidate_rows.empty:
        return []
    users_field = str(cfg.get("users_suspected_field", "suspected_fraud"))
    confirmed_field = str(cfg.get("confirmed_field", "confirmed_fraud"))
    confirmed_fraud_weight = float(cfg.get("confirmed_fraud_weight", 2.0))
    confirmed_not_fraud_weight = float(cfg.get("confirmed_not_fraud_weight", 2.0))

    candidate_entries: list[dict[str, Any]] = []
    member_ids: set[str] = set()
    for row in candidate_rows.to_dict("records"):
        draw_id = pd.to_numeric(row.get("draw_id"), errors="coerce")
        row_date = _source_alert_date(row)
        if pd.isna(draw_id) or row_date is None:
            continue
        for member_id, ccs_id in _candidate_member_map(row).items():
            member_ids.add(member_id)
            candidate_entries.append(
                {
                    "draw_id": int(draw_id),
                    "member_id": member_id,
                    "ccs_id": ccs_id,
                    "date": row_date,
                }
            )

    if not candidate_entries or not member_ids:
        return []

    users_collection = _collection_from_cfg(cfg, "users_collection_env_var", ENV_USERS_COLLECTION)
    user_docs = _find_docs_by_members(users_collection, member_ids)

    labels: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in candidate_entries:
        key = (entry["draw_id"], entry["member_id"])
        ccs_id = entry["ccs_id"]
        user_doc = user_docs.get(entry["member_id"])
        if user_doc:
            confirmed = _field_bool(user_doc, confirmed_field)
            periods = _suspected_periods(user_doc, users_field, entry["member_id"])
            if confirmed is not None and _date_in_periods(entry["date"], periods):
                label = "fraud" if confirmed else "not_fraud"
                labels[key] = {
                    "draw_id": entry["draw_id"],
                    "member_id": entry["member_id"],
                    "ccs_id": ccs_id,
                    "label": label,
                    "sample_weight": confirmed_fraud_weight if label == "fraud" else confirmed_not_fraud_weight,
                    "decided_at": _first_present(user_doc, "updatedAt", "updated_at"),
                    "label_source": "native_gk_users",
                }
    return list(labels.values())
