from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from fraud_detection.constants.constants import ENV_MONGODB_COLLECTION, ENV_MONGODB_DATABASE, ENV_MONGODB_URI, REPO_ROOT
from fraud_detection.utils.common import write_json
from fraud_detection.utils.mongodb import get_serving_mongo_collection


CCS_DAILY_PROFIT_COLUMNS = [
    "ccs_id",
    "member_id",
    "profit_date",
    "daily_profit",
    "daily_bet_amount",
    "daily_win_points",
    "draw_count",
]

_MISSING_ID_STRINGS = {"", "NAN", "NONE", "NULL", "NAT"}


@dataclass(frozen=True)
class CcsProfitDayResult:
    profit_date: date
    status: str
    rows_written: int = 0
    output_partitions: list[str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class CcsProfitBuildSummary:
    start_date: date
    end_date: date
    output_path: str
    days_processed: int
    days_succeeded: int
    skipped_days: int
    failed_days: int
    rows_written: int
    partitions_written: int
    force: bool
    elapsed_seconds: float
    status: str
    failures: list[dict[str, Any]]
    results: list[CcsProfitDayResult]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed_days else 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        for item in payload["results"]:
            item["profit_date"] = item["profit_date"].isoformat()
        return payload


def _utc_bound(value: date, *, end: bool = False) -> datetime:
    return datetime.combine(value, time.max if end else time.min, tzinfo=timezone.utc)


def _normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.upper() in _MISSING_ID_STRINGS:
        return None
    return text.upper()


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(object).where(series.notna(), None)
    return normalized.map(_normalize_identifier)


def _normalize_profit_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=CCS_DAILY_PROFIT_COLUMNS)
    if "_id" in frame.columns:
        id_frame = pd.json_normalize(frame["_id"])
        frame = pd.concat([frame.drop(columns=["_id"]), id_frame], axis=1)
    frame = frame.rename(
        columns={
            "bet_amount": "daily_bet_amount",
            "win_points": "daily_win_points",
            "draws": "draw_count",
        }
    )
    for column in CCS_DAILY_PROFIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0 if column not in {"ccs_id", "member_id", "profit_date"} else None
    frame["ccs_id"] = _normalize_identifier_series(frame["ccs_id"])
    frame["member_id"] = _normalize_identifier_series(frame["member_id"])
    frame["profit_date"] = pd.to_datetime(frame["profit_date"], errors="coerce").dt.date
    frame["daily_bet_amount"] = pd.to_numeric(frame["daily_bet_amount"], errors="coerce").fillna(0.0)
    frame["daily_win_points"] = pd.to_numeric(frame["daily_win_points"], errors="coerce").fillna(0.0)
    frame["daily_profit"] = pd.to_numeric(
        frame.get("daily_profit", frame["daily_win_points"] - frame["daily_bet_amount"]),
        errors="coerce",
    ).fillna(frame["daily_win_points"] - frame["daily_bet_amount"])
    frame["draw_count"] = pd.to_numeric(frame["draw_count"], errors="coerce").fillna(0).astype(int)
    return frame[CCS_DAILY_PROFIT_COLUMNS].dropna(subset=["ccs_id", "member_id", "profit_date"])


def aggregate_profit_documents(docs: Iterable[dict[str, Any]], *, timestamp_field: str = "trans_date") -> pd.DataFrame:
    rows = []
    for doc in docs:
        ts = pd.to_datetime(doc.get(timestamp_field), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        ccs_id = _normalize_identifier(doc.get("ccs_id")) or _normalize_identifier(doc.get("CCS_ID"))
        member_id = _normalize_identifier(doc.get("member_id")) or _normalize_identifier(doc.get("MEMBER_ID"))
        if not ccs_id or not member_id:
            continue
        bet = float(pd.to_numeric(pd.Series([doc.get("total_bet_amount", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        win = float(pd.to_numeric(pd.Series([doc.get("win_points", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        rows.append(
            {
                "ccs_id": ccs_id,
                "member_id": member_id,
                "profit_date": ts.date(),
                "daily_bet_amount": bet,
                "daily_win_points": win,
                "daily_profit": win - bet,
                "draw_id": doc.get("draw_id"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=CCS_DAILY_PROFIT_COLUMNS)
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["ccs_id", "member_id", "profit_date"], as_index=False).agg(
        daily_profit=("daily_profit", "sum"),
        daily_bet_amount=("daily_bet_amount", "sum"),
        daily_win_points=("daily_win_points", "sum"),
        draw_count=("draw_id", "nunique"),
    )
    return grouped[CCS_DAILY_PROFIT_COLUMNS]


def write_partitioned_profit(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    compression: str = "zstd",
    replace_dates: Iterable[date] | None = None,
) -> dict[str, Any]:
    base_path = Path(output_path)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    dates_to_replace = {pd.Timestamp(value).date() for value in (replace_dates or [])}
    out = frame.copy() if not frame.empty else pd.DataFrame(columns=CCS_DAILY_PROFIT_COLUMNS)
    out["profit_date"] = pd.to_datetime(out["profit_date"], errors="coerce").dt.date
    out = out.dropna(subset=["profit_date"])
    out["year"] = pd.to_datetime(out["profit_date"]).dt.year
    out["month"] = pd.to_datetime(out["profit_date"]).dt.month
    written_partitions = []
    rows_written = 0
    months = set()
    if not out.empty:
        months.update((int(year), int(month)) for year, month in out[["year", "month"]].drop_duplicates().itertuples(index=False, name=None))
    months.update((day.year, day.month) for day in dates_to_replace)
    for year, month in sorted(months):
        partition = base_path / f"year={int(year)}" / f"month={int(month):02d}"
        target = partition / "ccs_daily_profit.parquet"
        group = (
            out.loc[out["year"].eq(year) & out["month"].eq(month)]
            .drop(columns=["year", "month"])
            .copy()
            if not out.empty
            else pd.DataFrame(columns=CCS_DAILY_PROFIT_COLUMNS)
        )
        if group.empty and not target.exists():
            continue
        partition.mkdir(parents=True, exist_ok=True)
        merged = group.copy()
        if target.exists():
            existing = pd.read_parquet(target)
            existing["profit_date"] = pd.to_datetime(existing["profit_date"], errors="coerce").dt.date
            month_replace_dates = {day for day in dates_to_replace if day.year == year and day.month == month}
            if month_replace_dates:
                existing = existing.loc[~existing["profit_date"].isin(month_replace_dates)].copy()
            if not merged.empty:
                merged = pd.concat([existing, merged], ignore_index=True)
            else:
                merged = existing
        if not merged.empty:
            merged = merged.drop_duplicates(["ccs_id", "member_id", "profit_date"], keep="last")
        else:
            merged = pd.DataFrame(columns=CCS_DAILY_PROFIT_COLUMNS)
        merged.to_parquet(target, index=False, compression=compression)
        written_partitions.append(str(target))
        rows_written += len(group)
    return {"rows_written": int(rows_written), "partitions_written": written_partitions}


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _daily_profit_pipeline(day: date, *, timestamp_field: str) -> list[dict[str, Any]]:
    return [
        {
            "$match": {
                timestamp_field: {
                    "$gte": _utc_bound(day),
                    "$lte": _utc_bound(day, end=True),
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "ccs_id": "$ccs_id",
                    "member_id": "$member_id",
                    "profit_date": {"$dateToString": {"format": "%Y-%m-%d", "date": f"${timestamp_field}"}},
                },
                "daily_bet_amount": {"$sum": "$total_bet_amount"},
                "daily_win_points": {"$sum": "$win_points"},
                "draw_count": {"$addToSet": "$draw_id"},
            }
        },
        {
            "$project": {
                "_id": 1,
                "daily_bet_amount": 1,
                "daily_win_points": 1,
                "daily_profit": {"$subtract": ["$daily_win_points", "$daily_bet_amount"]},
                "draw_count": {"$size": "$draw_count"},
            }
        },
    ]


def _resolve_output_path(output_path: str | Path) -> Path:
    base_path = Path(output_path)
    return base_path if base_path.is_absolute() else REPO_ROOT / base_path


def _partition_file_for_day(output_path: str | Path, day: date) -> Path:
    return _resolve_output_path(output_path) / f"year={day.year}" / f"month={day.month:02d}" / "ccs_daily_profit.parquet"


def _has_existing_profit_day(output_path: str | Path, day: date) -> bool:
    target = _partition_file_for_day(output_path, day)
    if not target.exists():
        return False
    try:
        frame = pd.read_parquet(target, columns=["profit_date"])
    except Exception:
        return False
    if frame.empty:
        return False
    dates = pd.to_datetime(frame["profit_date"], errors="coerce").dt.date
    return bool(dates.eq(day).any())


def build_ccs_daily_profit(
    start_date: date,
    end_date: date,
    output_path: str,
    *,
    timestamp_field: str = "trans_date",
    compression: str = "zstd",
    uri_env_var: str = ENV_MONGODB_URI,
    database_env_var: str = ENV_MONGODB_DATABASE,
    collection_env_var: str = ENV_MONGODB_COLLECTION,
    force: bool = False,
    report_path: str | Path | None = None,
) -> CcsProfitBuildSummary:
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} must be on or before end_date {end_date}")
    started = time_module.perf_counter()
    collection = get_serving_mongo_collection(
        uri_env_var,
        database_env_var,
        collection_env_var,
    )
    results: list[CcsProfitDayResult] = []
    for day in _iter_dates(start_date, end_date):
        if not force and _has_existing_profit_day(output_path, day):
            results.append(CcsProfitDayResult(profit_date=day, status="skipped"))
            continue
        try:
            frame = _normalize_profit_frame(
                collection.aggregate(_daily_profit_pipeline(day, timestamp_field=timestamp_field), allowDiskUse=True)
            )
            write_result = write_partitioned_profit(
                frame,
                output_path,
                compression=compression,
                replace_dates=[day] if force else None,
            )
            results.append(
                CcsProfitDayResult(
                    profit_date=day,
                    status="succeeded",
                    rows_written=int(write_result["rows_written"]),
                    output_partitions=list(write_result["partitions_written"]),
                )
            )
        except Exception as exc:
            results.append(CcsProfitDayResult(profit_date=day, status="failed", error=str(exc)))

    failures = [
        {"profit_date": item.profit_date.isoformat(), "error": item.error}
        for item in results
        if item.status == "failed"
    ]
    summary = CcsProfitBuildSummary(
        start_date=start_date,
        end_date=end_date,
        output_path=str(_resolve_output_path(output_path)),
        days_processed=len(results),
        days_succeeded=sum(1 for item in results if item.status == "succeeded"),
        skipped_days=sum(1 for item in results if item.status == "skipped"),
        failed_days=sum(1 for item in results if item.status == "failed"),
        rows_written=sum(item.rows_written for item in results),
        partitions_written=len({part for item in results for part in (item.output_partitions or [])}),
        force=bool(force),
        elapsed_seconds=time_module.perf_counter() - started,
        status="FAILED" if failures else "FINISHED",
        failures=failures,
        results=results,
    )
    if report_path is not None:
        write_json(summary.to_dict(), Path(report_path))
    if failures:
        raise RuntimeError(f"CCS profit build failed for {len(failures)} day(s): {failures[:3]}")
    return summary
