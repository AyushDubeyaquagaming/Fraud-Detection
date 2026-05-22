from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, status

from fraud_detection.constants.constants import ENV_MONGODB_COLLECTION, ENV_MONGODB_DATABASE, ENV_MONGODB_URI, REPO_ROOT, RUNS_DIR
from fraud_detection.serving.dependencies import LiveScoringContext, get_live_scoring_context
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
from fraud_detection.serving.schemas import (
    AlertDraw,
    AlertFlaggedMember,
    AlertDrawResponse,
    CcsScore,
    CcsScoreRequest,
    CcsScoreResponse,
    DrawScoreResponse,
    DrawPayloadScoreRequest,
    EvidenceDraw,
    ErrorResponse,
    MemberScoreResponse,
)
from fraud_detection.utils.mongodb import get_serving_mongo_collection

router = APIRouter(tags=["live-scoring"])
DEFAULT_CANDIDATE_STORE_PATH = REPO_ROOT / "data_store" / "candidate_draws"
CURRENT_BACKFILL_PATH = REPO_ROOT / "artifacts" / "current" / "live_predictions_backfill.parquet"


def _raw_collection():
    return get_serving_mongo_collection(
        ENV_MONGODB_URI,
        ENV_MONGODB_DATABASE,
        ENV_MONGODB_COLLECTION,
    )


def _make_scorer(context: LiveScoringContext) -> DrawScorer:
    return DrawScorer(
        context.model_bundle,
        source_run_id=context.source_run_id or context.model_bundle.get("source_run_id"),
        partnership_table=context.partnership_table,
        ccs_concentration_table=getattr(context, "ccs_concentration_table", pd.DataFrame()),
    )


def _member_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _finite_float(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    try:
        if pd.isna(number):
            return default
    except (TypeError, ValueError):
        return default
    return float(number)


def _alert_window_bounds(lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.now(tz="UTC")
    start = (end - pd.Timedelta(days=int(lookback_days))).normalize()
    return start, end


def _public_response_details(details: list[Any] | None, context: LiveScoringContext) -> list[str]:
    normalized: list[str] = []
    for item in details or []:
        value = str(item).strip()
        if value and value not in normalized:
            normalized.append(value)
    label_status = str(context.evaluation_metadata.get("label_status", "unknown")).strip().lower()
    if label_status != "available" and "stage2_unavailable_no_labels" not in normalized:
        normalized.append("stage2_unavailable_no_labels")
    return normalized


def _window_query(timestamp_field: str, lookback_days: int) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    return {timestamp_field: {"$gte": start, "$lte": end}}


def _draw_ids_for_member(collection, member_id: str, timestamp_field: str, lookback_days: int, max_draws: int) -> list[int]:
    member_variants = [member_id, member_id.lower(), member_id.upper()]
    query = {"member_id": {"$in": member_variants}, **_window_query(timestamp_field, lookback_days)}
    draw_ids = collection.distinct("draw_id", query)
    if not draw_ids:
        query = {"member_id": {"$in": member_variants}}
        draw_ids = collection.distinct("draw_id", query)
    return sorted(int(draw_id) for draw_id in draw_ids if draw_id is not None)[-max_draws:]


def _candidate_store_path(context: LiveScoringContext) -> Path:
    raw_path = context.model_bundle.get("candidate_store_path") or context.model_bundle.get("candidate_draws_path")
    return Path(raw_path) if raw_path and Path(raw_path).is_absolute() else (REPO_ROOT / raw_path if raw_path else DEFAULT_CANDIDATE_STORE_PATH)


def _candidate_rows_for_member(
    member_id: str,
    *,
    context: LiveScoringContext,
    max_draws: int,
    draw_id: int | None = None,
) -> pd.DataFrame:
    store_path = _candidate_store_path(context)
    if not store_path.exists():
        return pd.DataFrame()

    def has_member(values: Any) -> bool:
        if values is None:
            return False
        try:
            members = list(values)
        except TypeError:
            return False
        return member_id in {_member_key(value) for value in members if str(value).strip()}

    dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
    filter_expr = ds.field("draw_id") == int(draw_id) if draw_id is not None else None
    matches: list[pd.DataFrame] = []
    for batch in dataset.scanner(filter=filter_expr, batch_size=20_000).to_batches():
        frame = batch.to_pandas()
        if frame.empty or "member_ids" not in frame.columns:
            continue
        mask = frame["member_ids"].apply(has_member)
        if mask.any():
            matches.append(frame.loc[mask].copy())
    if not matches:
        return pd.DataFrame()
    out = pd.concat(matches, ignore_index=True)
    if "trans_date_min" in out.columns:
        out = out.sort_values("trans_date_min")
    return out.tail(max_draws)


def _candidate_rows_for_draw(draw_id: int, *, context: LiveScoringContext) -> pd.DataFrame:
    store_path = _candidate_store_path(context)
    if not store_path.exists():
        return pd.DataFrame()
    dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
    filter_expr = ds.field("draw_id") == int(draw_id)
    try:
        fragments = sorted(
            list(dataset.get_fragments()),
            key=lambda fragment: str(getattr(fragment, "path", "")),
            reverse=True,
        )
        for fragment in fragments:
            table = fragment.to_table(filter=filter_expr)
            if table.num_rows:
                return table.to_pandas()
        return pd.DataFrame()
    except Exception:
        table = dataset.to_table(filter=filter_expr)
        return table.to_pandas() if table.num_rows else pd.DataFrame()


def _candidate_rows_for_window(
    *,
    context: LiveScoringContext,
    lookback_days: int,
    max_draws: int,
) -> pd.DataFrame:
    store_path = _candidate_store_path(context)
    if not store_path.exists():
        return pd.DataFrame()
    start, end = _alert_window_bounds(lookback_days)
    dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
    matches: list[pd.DataFrame] = []

    filter_expr = None
    if "trans_date_min" in set(dataset.schema.names):
        filter_expr = (ds.field("trans_date_min") >= start.to_pydatetime()) & (
            ds.field("trans_date_min") <= end.to_pydatetime()
        )

    def scan_batches(active_filter):
        scanner_args: dict[str, Any] = {"batch_size": 20_000}
        if active_filter is not None:
            scanner_args["filter"] = active_filter
        return dataset.scanner(**scanner_args).to_batches()

    def trim_matches() -> None:
        if not matches:
            return
        out = pd.concat(matches, ignore_index=True)
        out["_sort_date"] = pd.to_datetime(out["trans_date_min"], errors="coerce", utc=True)
        out = out.sort_values(["_sort_date", "draw_id"]).drop(columns=["_sort_date"])
        matches.clear()
        matches.append(out.tail(int(max_draws)).copy())

    def collect(active_filter) -> None:
        matched_rows = sum(len(item) for item in matches)
        for batch in scan_batches(active_filter):
            frame = batch.to_pandas()
            if frame.empty or "trans_date_min" not in frame.columns:
                continue
            draw_dates = pd.to_datetime(frame["trans_date_min"], errors="coerce", utc=True)
            mask = draw_dates.ge(start) & draw_dates.le(end)
            if mask.any():
                chunk = frame.loc[mask].copy()
                matches.append(chunk)
                matched_rows += len(chunk)
            if matched_rows > int(max_draws) * 2:
                trim_matches()
                matched_rows = sum(len(item) for item in matches)

    try:
        collect(filter_expr)
    except Exception:
        matches.clear()
        collect(None)
    if not matches:
        return pd.DataFrame()
    out = pd.concat(matches, ignore_index=True)
    out["_sort_date"] = pd.to_datetime(out["trans_date_min"], errors="coerce", utc=True)
    out = out.sort_values(["_sort_date", "draw_id"]).drop(columns=["_sort_date"])
    return out.tail(int(max_draws))


def _candidate_member_to_ccs(row: dict[str, Any] | pd.Series) -> dict[str, str]:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    members = [_member_key(value) for value in source.get("member_ids", [])]
    ccs_values = list(source.get("ccs_ids", []))
    mapping: dict[str, str] = {}
    for idx, member_id in enumerate(members):
        if not member_id:
            continue
        ccs_value = ccs_values[idx] if idx < len(ccs_values) else None
        ccs_id = str(ccs_value).strip().upper() if ccs_value is not None and str(ccs_value).strip() else "UNKNOWN"
        mapping[member_id] = ccs_id
    return mapping


def _candidate_draw_date(row: dict[str, Any] | pd.Series) -> str | None:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    draw_date = pd.to_datetime(source.get("trans_date_min"), errors="coerce", utc=True)
    return None if pd.isna(draw_date) else draw_date.isoformat()


def _flagged_members_from_doc(result_doc: dict[str, Any]) -> set[str]:
    members = {_member_key(item.get("member_id")) for item in result_doc.get("flagged_members", [])}
    for partnership in result_doc.get("partnerships", []):
        members.update(_member_key(member) for member in partnership.get("member_ids", []))
    members.discard("")
    return members


def _partnership_partner_lookup(result_doc: dict[str, Any]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    for partnership in result_doc.get("partnerships", []):
        members = [_member_key(member) for member in partnership.get("member_ids", [])]
        members = [member for member in members if member]
        for member_id in members:
            lookup.setdefault(member_id, set()).update(member for member in members if member != member_id)
    return lookup


def _enrich_result_doc_amounts(result_doc: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    result_doc = result_doc.copy()
    amount_context = DrawScorer._candidate_amount_context(row)
    partner_lookup = _partnership_partner_lookup(result_doc)
    flagged_by_member = {
        _member_key(item.get("member_id")): dict(item)
        for item in result_doc.get("flagged_members", [])
    }
    for member_id in _flagged_members_from_doc(result_doc):
        current = flagged_by_member.get(member_id, {"member_id": member_id})
        current["stage1_score_in_draw"] = float(current.get("stage1_score_in_draw") or 0.0)
        current["stage2_score"] = float(current.get("stage2_score") or 0.0)
        if not current.get("best_partner_member_id") and partner_lookup.get(member_id):
            current["best_partner_member_id"] = sorted(partner_lookup[member_id])[0]
        flagged_by_member[member_id] = DrawScorer._enrich_flagged_member_amounts(current, amount_context)
    result_doc["flagged_members"] = list(flagged_by_member.values())
    return result_doc


def _prediction_backfill_path(context: LiveScoringContext) -> Path | None:
    for key in ("live_predictions_path", "live_predictions_backfill_path", "batch_predictions_path"):
        raw_path = context.model_bundle.get(key)
        if raw_path:
            path = Path(raw_path)
            path = path if path.is_absolute() else REPO_ROOT / path
            if path.exists():
                return path
    if CURRENT_BACKFILL_PATH.exists():
        return CURRENT_BACKFILL_PATH
    paths = sorted(
        RUNS_DIR.glob("*/model_training/live_predictions_backfill.parquet"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


@lru_cache(maxsize=2)
def _prediction_backfill_frame(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@lru_cache(maxsize=4)
def _prediction_backfill_recent_frame(path_str: str, mtime_ns: int, limit: int) -> pd.DataFrame:
    limit = max(1, int(limit))
    parquet_file = pq.ParquetFile(path_str)
    if parquet_file.metadata.num_rows <= limit:
        return pd.read_parquet(path_str)

    selected_groups: list[int] = []
    rows_remaining = limit
    for group_idx in range(parquet_file.num_row_groups - 1, -1, -1):
        selected_groups.append(group_idx)
        rows_remaining -= parquet_file.metadata.row_group(group_idx).num_rows
        if rows_remaining <= 0:
            break
    table = parquet_file.read_row_groups(sorted(selected_groups))
    return table.to_pandas().tail(limit)


def _cached_prediction_docs(candidate_rows: pd.DataFrame, context: LiveScoringContext) -> dict[int, dict[str, Any]]:
    path = _prediction_backfill_path(context)
    if path is None or candidate_rows.empty or "draw_id" not in candidate_rows.columns:
        return {}
    draw_ids = set(pd.to_numeric(candidate_rows["draw_id"], errors="coerce").dropna().astype(int).tolist())
    if not draw_ids:
        return {}
    predictions = _prediction_backfill_frame(str(path), path.stat().st_mtime_ns)
    if predictions.empty or "draw_id" not in predictions.columns:
        return {}
    predictions = predictions.loc[pd.to_numeric(predictions["draw_id"], errors="coerce").isin(draw_ids)].copy()
    docs: dict[int, dict[str, Any]] = {}
    for item in predictions.to_dict("records"):
        if item.get("draw_id") is None:
            continue
        docs[int(item["draw_id"])] = item
    return docs


def _prediction_backfill_docs(context: LiveScoringContext, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = _prediction_backfill_path(context)
    if path is None:
        return []
    if limit is not None:
        predictions = _prediction_backfill_recent_frame(str(path), path.stat().st_mtime_ns, max(1, int(limit)))
    else:
        predictions = _prediction_backfill_frame(str(path), path.stat().st_mtime_ns)
    if predictions.empty:
        return []
    docs: list[dict[str, Any]] = []
    for item in predictions.to_dict("records"):
        draw_id = pd.to_numeric(item.get("draw_id"), errors="coerce")
        if pd.isna(draw_id):
            continue
        item["draw_id"] = int(draw_id)
        docs.append(item)
    return docs


def _doc_draw_timestamp(result_doc: dict[str, Any], row: dict[str, Any] | None) -> pd.Timestamp | None:
    if row is not None:
        candidate_ts = pd.to_datetime(row.get("trans_date_min"), errors="coerce", utc=True)
        if not pd.isna(candidate_ts):
            return candidate_ts
    for key in ("draw_date", "trans_date_min"):
        value = pd.to_datetime(result_doc.get(key), errors="coerce", utc=True)
        if not pd.isna(value):
            return value
    return None


def _candidate_row_for_doc(draw_id: int, context: LiveScoringContext) -> dict[str, Any] | None:
    candidate_rows = _candidate_rows_for_draw(draw_id, context=context)
    if candidate_rows.empty:
        return None
    return candidate_rows.iloc[0].to_dict()


def _build_alert_draws_from_backfill(
    *,
    context: LiveScoringContext,
    limit: int,
    lookback_days: int,
    max_draws: int,
    min_bet_amount: float | None,
    ccs_ids: set[str] | None = None,
) -> tuple[list[AlertDraw], int]:
    backend_limit = max(1, int(max_draws))
    docs = _prediction_backfill_docs(context, limit=backend_limit)
    start, end = _alert_window_bounds(lookback_days)
    requested_ccs = {_member_key(value) for value in (ccs_ids or set()) if str(value).strip()}
    docs_by_draw_id: dict[int, dict[str, Any]] = {}
    for result_doc in docs:
        draw_id = pd.to_numeric(result_doc.get("draw_id"), errors="coerce")
        if not pd.isna(draw_id):
            docs_by_draw_id[int(draw_id)] = result_doc

    candidates: list[tuple[pd.Timestamp, AlertDraw]] = []

    def append_alert(result_doc: dict[str, Any], row: dict[str, Any] | None) -> bool:
        draw_id = pd.to_numeric(result_doc.get("draw_id"), errors="coerce")
        if pd.isna(draw_id):
            return False
        draw_id = int(draw_id)
        flagged_members = sorted(_flagged_members_from_doc(result_doc))
        if not result_doc.get("requires_review") or not flagged_members:
            return False
        draw_ts = _doc_draw_timestamp(result_doc, row)
        if draw_ts is None or draw_ts < start or draw_ts > end:
            return False
        member_to_ccs = _candidate_member_to_ccs(row) if row is not None else {}
        partner_lookup = _partnership_partner_lookup(result_doc)
        flagged_by_member = {
            _member_key(item.get("member_id")): dict(item)
            for item in result_doc.get("flagged_members", [])
        }
        alert_members: list[AlertFlaggedMember] = []
        for member_id in flagged_members:
            member_ccs_id = member_to_ccs.get(member_id, "UNKNOWN")
            if requested_ccs and _member_key(member_ccs_id) not in requested_ccs:
                continue
            item = flagged_by_member.get(member_id, {"member_id": member_id})
            partners = set(partner_lookup.get(member_id, set()))
            best_partner = item.get("best_partner_member_id")
            if best_partner:
                partners.add(_member_key(best_partner))
            alert_members.append(
                AlertFlaggedMember(
                    member_id=member_id,
                    ccsId=member_ccs_id,
                    stage1_score_in_draw=_finite_float(item.get("stage1_score_in_draw")),
                    best_partner_member_id=item.get("best_partner_member_id"),
                    betAmount=_finite_float(item.get("bet_amount", item.get("betAmount"))),
                    winAmount=_finite_float(item.get("win_amount", item.get("winAmount"))),
                    highAmountFlag=bool(item.get("high_amount_flag") or item.get("highAmountFlag")),
                    highAmountReason=item.get("high_amount_reason") or item.get("highAmountReason"),
                    partner_member_ids=sorted(partners),
                )
            )
        if requested_ccs and not alert_members:
            return False
        if min_bet_amount is not None and not any(float(item.bet_amount or 0.0) >= min_bet_amount for item in alert_members):
            return False
        high_amount_member_count = sum(1 for item in alert_members if item.high_amount_flag)
        alert = AlertDraw(
            draw_id=draw_id,
            draw_date=_candidate_draw_date(row) if row is not None else draw_ts.isoformat(),
            risk_tier="HIGH",
            partnership_count=len(result_doc.get("partnerships", [])),
            flagged_member_count=len(alert_members),
            flagged_member_ids=sorted({member.member_id for member in alert_members}),
            flaggedMembers=alert_members,
            ccs_ids=sorted({_member_key(member.ccs_id) or "UNKNOWN" for member in alert_members}),
            highAmountMemberCount=high_amount_member_count,
            maxBetAmount=max((float(item.bet_amount or 0.0) for item in alert_members), default=0.0),
            maxWinAmount=max((float(item.win_amount or 0.0) for item in alert_members), default=0.0),
            max_stage1_score=float(result_doc.get("max_stage1_score") or 0.0),
            response_details=_public_response_details(result_doc.get("response_details", []), context),
        )
        candidates.append((draw_ts, alert))
        return True

    matched_candidate_docs = 0
    scanned_docs = 0
    try:
        candidate_rows = _candidate_rows_for_window(
            context=context,
            lookback_days=lookback_days,
            max_draws=backend_limit,
        )
    except Exception:
        candidate_rows = pd.DataFrame()
    if not candidate_rows.empty:
        for row in candidate_rows.to_dict("records"):
            draw_id = pd.to_numeric(row.get("draw_id"), errors="coerce")
            if pd.isna(draw_id):
                continue
            result_doc = docs_by_draw_id.get(int(draw_id))
            if result_doc is None:
                continue
            matched_candidate_docs += 1
            scanned_docs += 1
            append_alert(result_doc, row)
            if matched_candidate_docs >= backend_limit:
                break

    if matched_candidate_docs == 0:
        for result_doc in docs[:backend_limit]:
            scanned_docs += 1
            draw_id = pd.to_numeric(result_doc.get("draw_id"), errors="coerce")
            row = None if pd.isna(draw_id) else _candidate_row_for_doc(int(draw_id), context)
            append_alert(result_doc, row)
            if len(candidates) >= backend_limit:
                break

    candidates.sort(key=lambda item: (item[0], item[1].draw_id), reverse=True)
    alerts = [item[1] for item in candidates[:backend_limit]]
    alerts.sort(
        key=lambda item: (
            item.high_amount_member_count,
            item.max_win_amount,
            item.max_bet_amount,
            item.draw_date or "",
            item.draw_id,
        ),
        reverse=True,
    )
    return alerts[:limit], scanned_docs


def _candidate_result_docs(
    candidate_rows: pd.DataFrame,
    scorer: DrawScorer,
    *,
    context: LiveScoringContext | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if candidate_rows.empty:
        return []
    row_dicts = candidate_rows.to_dict("records")
    cached_docs = _cached_prediction_docs(candidate_rows, context) if context is not None else {}
    docs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in row_dicts:
        cached = cached_docs.get(int(row["draw_id"]))
        if cached is None:
            missing_rows.append(row)
            continue
        cached = _enrich_result_doc_amounts(cached, row)
        docs.append((cached, row))
    if missing_rows:
        missing_frame = pd.DataFrame(missing_rows)
        results = scorer.score_candidate_batch(missing_frame)
        for row, result in zip(missing_rows, results):
            result_doc = _enrich_result_doc_amounts(result.to_mongo_doc(), row)
            docs.append((result_doc, row))
    return docs


def _member_evidence(result_doc: dict[str, Any], member_id: str) -> EvidenceDraw | None:
    partners: set[str] = set()
    max_union_coverage = None
    for partnership in result_doc.get("partnerships", []):
        members = {_member_key(value) for value in partnership.get("member_ids", [])}
        if member_id not in members:
            continue
        partners.update(member for member in members if member != member_id)
        union = partnership.get("union_coverage")
        if union is not None:
            max_union_coverage = max(float(union), max_union_coverage or 0.0)

    flagged = None
    for item in result_doc.get("flagged_members", []):
        if _member_key(item.get("member_id")) == member_id:
            flagged = item
            partner = item.get("best_partner_member_id")
            if partner:
                partners.add(_member_key(partner))
            break

    if not partners and flagged is None:
        return None
    return EvidenceDraw(
        draw_id=int(result_doc["draw_id"]),
        score_reason="partnership_pattern" if partners else "model_score",
        partner_member_ids=sorted(partners),
        stage1_score_in_draw=float(flagged["stage1_score_in_draw"]) if flagged else None,
        stage2_score=float(flagged["stage2_score"]) if flagged else None,
        max_union_coverage=max_union_coverage,
    )


def _rows_for_draw_id(draw_id: int) -> list[dict[str, Any]]:
    try:
        collection = _raw_collection()
        return list(collection.find({"draw_id": int(draw_id)}, {"_id": 0}))
    except Exception:
        return []


def _draw_response_from_rows(draw_id: int, rows: list[dict[str, Any]], context: LiveScoringContext) -> DrawScoreResponse:
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No raw rows found for draw_id={draw_id}",
        )
    scorer = _make_scorer(context)
    result = scorer.score_draw(pd.DataFrame(rows))
    return DrawScoreResponse(**result.to_mongo_doc())


def _draw_response_by_id(draw_id: int, context: LiveScoringContext) -> DrawScoreResponse:
    scorer = _make_scorer(context)
    candidate_rows = _candidate_rows_for_draw(draw_id, context=context)
    if not candidate_rows.empty:
        result = scorer.score_candidate_draw(candidate_rows.iloc[0])
        return DrawScoreResponse(**result.to_mongo_doc())
    return _draw_response_from_rows(draw_id, _rows_for_draw_id(draw_id), context)


@router.post(
    "/score/draw/payload",
    response_model=DrawScoreResponse,
    include_in_schema=False,
)
def score_draw_payload(
    request: DrawPayloadScoreRequest,
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> DrawScoreResponse:
    rows = [
        {
            "draw_id": int(request.draw_id),
            "member_id": player.member_id,
            "ccs_id": player.ccs_id,
            "total_bet_amount": player.total_bet_amount,
            "win_points": player.win_points,
            "bets": [bet.model_dump() for bet in player.bets],
            "trans_date": request.trans_date,
        }
        for player in request.players
    ]
    scorer = _make_scorer(context)
    result = scorer.score_draw(pd.DataFrame(rows))
    return DrawScoreResponse(**result.to_mongo_doc())


@router.get(
    "/score/draw/{draw_id}",
    response_model=DrawScoreResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def score_draw_by_id(
    draw_id: int,
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> DrawScoreResponse:
    return _draw_response_by_id(int(draw_id), context)


@router.get(
    "/score/member/{member_id}",
    response_model=MemberScoreResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def score_member_by_id(
    member_id: str,
    lookback_days: int = 30,
    max_draws: int = 500,
    draw_id: int | None = None,
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> MemberScoreResponse:
    return _score_member_impl(
        _member_key(member_id),
        lookback_days=lookback_days,
        max_draws=max_draws,
        draw_id=draw_id,
        context=context,
    )


def _score_member_impl(
    member_id: str,
    *,
    lookback_days: int,
    max_draws: int,
    draw_id: int | None,
    context: LiveScoringContext,
) -> MemberScoreResponse:
    exact_draw_id = int(draw_id) if draw_id is not None else None
    scorer = _make_scorer(context)
    evidence: list[EvidenceDraw] = []

    candidate_rows = _candidate_rows_for_member(member_id, context=context, max_draws=max_draws, draw_id=exact_draw_id)
    if not candidate_rows.empty:
        for result_doc, _ in _candidate_result_docs(candidate_rows, scorer, context=context):
            item = _member_evidence(result_doc, member_id)
            if item is not None:
                evidence.append(item)

    if evidence:
        return MemberScoreResponse(
            member_id=member_id,
            risk_tier="HIGH",
            lookback_days=lookback_days,
            draws_scanned=len(candidate_rows),
            evidence_draws=evidence,
        )

    if exact_draw_id is not None:
        collection = None
        draw_ids = [exact_draw_id]
    else:
        try:
            collection = _raw_collection()
            draw_ids = _draw_ids_for_member(
                collection, member_id, context.timestamp_field, lookback_days, max_draws
            )
        except Exception:
            collection = None
            draw_ids = []

    for current_draw_id in draw_ids:
        if collection is None:
            continue
        try:
            rows = list(collection.find({"draw_id": current_draw_id}, {"_id": 0}))
        except Exception:
            rows = []
        if not rows:
            continue
        result_doc = scorer.score_draw(pd.DataFrame(rows)).to_mongo_doc()
        item = _member_evidence(result_doc, member_id)
        if item is not None:
            evidence.append(item)

    if not draw_ids and not evidence:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No draws found for member_id={member_id}")

    return MemberScoreResponse(
        member_id=member_id,
        risk_tier="HIGH" if evidence else "LOW",
        lookback_days=lookback_days,
        draws_scanned=max(len(draw_ids), len(evidence)),
        evidence_draws=evidence,
    )


@router.post(
    "/score/ccs",
    response_model=CcsScoreResponse,
    responses={503: {"model": ErrorResponse}},
)
def score_ccs(
    request: CcsScoreRequest,
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> CcsScoreResponse:
    requested = {_member_key(value) for value in request.ccs_ids or [] if str(value).strip()}
    alerts, draws_scanned = _build_alert_draws_from_backfill(
        context=context,
        limit=max(1, request.max_draws),
        lookback_days=request.lookback_days,
        max_draws=request.max_draws,
        min_bet_amount=None,
        ccs_ids=requested,
    )
    ccs_map: dict[str, dict[str, Any]] = {ccs_id: {"members": set(), "draw_ids": set(), "evidence": []} for ccs_id in requested}
    for alert in alerts:
        for member in alert.flagged_members:
            ccs_id = _member_key(member.ccs_id)
            if not ccs_id or (requested and ccs_id not in requested):
                continue
            entry = ccs_map.setdefault(ccs_id, {"members": set(), "draw_ids": set(), "evidence": []})
            entry["members"].add(member.member_id)
            entry["draw_ids"].add(int(alert.draw_id))
            entry["evidence"].append(
                {
                    "draw_id": int(alert.draw_id),
                    "member_id": member.member_id,
                    "best_partner_member_id": member.best_partner_member_id,
                    "stage1_score_in_draw": float(member.stage1_score_in_draw),
                    "bet_amount": float(member.bet_amount or 0.0),
                    "win_amount": float(member.win_amount or 0.0),
                    "high_amount_flag": bool(member.high_amount_flag),
                }
            )

    scores = [
        CcsScore(
            ccs_id=ccs_id,
            risk_tier="HIGH" if values["members"] else "LOW",
            flagged_member_count=len(values["members"]),
            flagged_members=sorted(values["members"]),
            evidence_draw_ids=sorted(values["draw_ids"]),
            evidence=sorted(
                values["evidence"],
                key=lambda item: (item["high_amount_flag"], item["win_amount"], item["bet_amount"], item["draw_id"]),
                reverse=True,
            )[:25],
        )
        for ccs_id, values in ccs_map.items()
    ]
    scores.sort(key=lambda item: (item.flagged_member_count, len(item.evidence_draw_ids), item.ccs_id), reverse=True)
    return CcsScoreResponse(
        lookback_days=request.lookback_days,
        draws_scanned=draws_scanned,
        ccs_scores=scores,
    )


@router.get(
    "/score/alerts",
    response_model=AlertDrawResponse,
    responses={503: {"model": ErrorResponse}},
)
def score_alerts(
    lookback_days: int = 7,
    max_draws: int = 10000,
    limit: int = 250,
    betAmount: float | None = None,
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> AlertDrawResponse:
    lookback_days = max(1, min(int(lookback_days), 30))
    max_draws = max(1, min(int(max_draws), 50_000))
    limit = max(1, min(int(limit), 1_000))
    min_bet_amount = None if betAmount is None else max(0.0, float(betAmount))
    alerts, draws_scanned = _build_alert_draws_from_backfill(
        context=context,
        limit=limit,
        lookback_days=lookback_days,
        max_draws=max_draws,
        min_bet_amount=min_bet_amount,
    )
    return AlertDrawResponse(
        lookback_days=lookback_days,
        draws_scanned=draws_scanned,
        alert_draw_count=len(alerts),
        alerts=alerts,
    )
