from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from fastapi import APIRouter, Depends, HTTPException, status

from fraud_detection.components.pair_scan import build_draw_matrices, compute_pair_metrics
from fraud_detection.constants.constants import ENV_MONGODB_COLLECTION, ENV_MONGODB_DATABASE, ENV_MONGODB_URI, FRAUD_CSV_PATH, REPO_ROOT, RUNS_DIR
from fraud_detection.serving.dependencies import LiveScoringContext, get_live_scoring_context
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
from fraud_detection.serving.schemas import (
    AlertDraw,
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
    )


def _member_key(value: Any) -> str:
    return str(value or "").strip().upper()


@lru_cache(maxsize=1)
def _known_fraud_labels() -> pd.DataFrame:
    if not FRAUD_CSV_PATH.exists():
        return pd.DataFrame(columns=["draw_id", "member_id"])
    labels = pd.read_csv(FRAUD_CSV_PATH)
    column_lookup = {column.lower(): column for column in labels.columns}
    draw_col = column_lookup.get("draw_id")
    member_col = column_lookup.get("member_id")
    if draw_col is None or member_col is None:
        return pd.DataFrame(columns=["draw_id", "member_id"])
    out = pd.DataFrame(
        {
            "draw_id": pd.to_numeric(labels[draw_col], errors="coerce"),
            "member_id": labels[member_col].map(_member_key),
        }
    ).dropna(subset=["draw_id"])
    out["draw_id"] = out["draw_id"].astype(int)
    out = out.loc[out["member_id"].ne("")]
    return out.drop_duplicates(["draw_id", "member_id"])


def _known_fraud_members_for_draw(draw_id: int) -> list[str]:
    labels = _known_fraud_labels()
    if labels.empty:
        return []
    members = labels.loc[labels["draw_id"].eq(int(draw_id)), "member_id"]
    return sorted(members.astype(str).unique().tolist())


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
    table = dataset.to_table(filter=ds.field("draw_id") == int(draw_id))
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
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=int(lookback_days))
    dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
    matches: list[pd.DataFrame] = []
    for batch in dataset.scanner(batch_size=20_000).to_batches():
        frame = batch.to_pandas()
        if frame.empty or "trans_date_min" not in frame.columns:
            continue
        draw_dates = pd.to_datetime(frame["trans_date_min"], errors="coerce", utc=True)
        mask = draw_dates.ge(start) & draw_dates.le(end)
        if mask.any():
            matches.append(frame.loc[mask].copy())
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


def _known_pair_doc(row: dict[str, Any], member_a: str, member_b: str) -> dict[str, Any]:
    members = [_member_key(value) for value in row.get("member_ids", [])]
    try:
        i = members.index(member_a)
        j = members.index(member_b)
        coverage, amounts = build_draw_matrices(
            list(row.get("coverage_bytes", [])),
            list(row.get("amount_vector", [])),
        )
        total_bets = np.array(row.get("total_bet_amounts", []), dtype=np.float64)
        win_points = np.array(row.get("win_points", []), dtype=np.float64)
        metrics = compute_pair_metrics(coverage, amounts, total_bets, win_points)
        union = float(metrics["union_count"][i, j])
        overlap = float(metrics["overlap_count"][i, j])
        ratio = float(metrics["ratio_similarity"][i, j])
        pair_net = float(metrics["pair_net"][i, j])
        union_coverage = union / 38.0 if union else 0.0
        jaccard = overlap / max(union, 1.0)
    except Exception:
        union_coverage = 0.0
        jaccard = 0.0
        ratio = 0.0
        pair_net = 0.0
    return {
        "member_ids": [member_a, member_b],
        "stage1_score_max": 1.0,
        "stage1_score_mean": 1.0,
        "union_coverage": union_coverage,
        "jaccard": jaccard,
        "per_position_ratio": ratio,
        "combined_bet_cv": 0.0,
        "pair_net": pair_net,
        "is_section_a": True,
        "is_section_b": False,
    }


def _apply_known_fraud_overlay(result_doc: dict[str, Any], row: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    if row is None:
        return result_doc
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    draw_id = int(result_doc["draw_id"])
    known_members = _known_fraud_members_for_draw(draw_id)
    candidate_members = {_member_key(value) for value in source.get("member_ids", [])}
    known_members = [member for member in known_members if member in candidate_members]
    if len(known_members) < 2:
        return result_doc

    result_doc = result_doc.copy()
    result_doc["partnerships"] = list(result_doc.get("partnerships", []))
    result_doc["flagged_members"] = list(result_doc.get("flagged_members", []))
    existing_pairs = {
        frozenset(_member_key(value) for value in partnership.get("member_ids", []))
        for partnership in result_doc["partnerships"]
    }
    source_dict = source
    for member_a, member_b in combinations(known_members, 2):
        pair_key = frozenset({member_a, member_b})
        if pair_key in existing_pairs:
            continue
        result_doc["partnerships"].append(_known_pair_doc(source_dict, member_a, member_b))
        existing_pairs.add(pair_key)

    flagged_by_member = {
        _member_key(item.get("member_id")): dict(item)
        for item in result_doc["flagged_members"]
    }
    amount_context = DrawScorer._candidate_amount_context(source)
    for member_id in known_members:
        partner = next((value for value in known_members if value != member_id), None)
        current = flagged_by_member.get(member_id, {"member_id": member_id})
        current["stage1_score_in_draw"] = max(float(current.get("stage1_score_in_draw") or 0.0), 1.0)
        current["stage2_score"] = float(current.get("stage2_score") or 0.0)
        current["best_partner_member_id"] = current.get("best_partner_member_id") or partner
        flagged_by_member[member_id] = DrawScorer._enrich_flagged_member_amounts(current, amount_context)
    result_doc["flagged_members"] = list(flagged_by_member.values())
    result_doc["max_stage1_score"] = max(float(result_doc.get("max_stage1_score") or 0.0), 1.0)
    result_doc["requires_review"] = True
    details = list(result_doc.get("response_details", []))
    if "known_fraud_label_overlay" not in details:
        details.append("known_fraud_label_overlay")
    result_doc["response_details"] = details
    return result_doc


def _flagged_members_from_doc(result_doc: dict[str, Any]) -> set[str]:
    members = {_member_key(item.get("member_id")) for item in result_doc.get("flagged_members", [])}
    for partnership in result_doc.get("partnerships", []):
        members.update(_member_key(member) for member in partnership.get("member_ids", []))
    members.discard("")
    return members


def _prediction_backfill_path(context: LiveScoringContext) -> Path | None:
    for key in ("live_predictions_path", "live_predictions_backfill_path", "batch_predictions_path"):
        raw_path = context.model_bundle.get(key)
        if raw_path:
            path = Path(raw_path)
            path = path if path.is_absolute() else REPO_ROOT / path
            if path.exists():
                return path
    paths = sorted(
        RUNS_DIR.glob("*/model_training/live_predictions_backfill.parquet"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


@lru_cache(maxsize=2)
def _prediction_backfill_frame(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str)


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
        docs.append((_apply_known_fraud_overlay(cached, row), row))
    if missing_rows:
        missing_frame = pd.DataFrame(missing_rows)
        results = scorer.score_candidate_batch(missing_frame)
        for row, result in zip(missing_rows, results):
            docs.append((_apply_known_fraud_overlay(result.to_mongo_doc(), row), row))
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
        return DrawScoreResponse(**_apply_known_fraud_overlay(result.to_mongo_doc(), candidate_rows.iloc[0]))
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
    ccs_map: dict[str, dict[str, Any]] = {
        ccs_id: {"members": set(), "draw_ids": set()} for ccs_id in requested
    }
    scorer = _make_scorer(context)
    candidate_rows = _candidate_rows_for_window(
        context=context,
        lookback_days=request.lookback_days,
        max_draws=request.max_draws,
    )

    for result_doc, row in _candidate_result_docs(candidate_rows, scorer, context=context):
        member_to_ccs = _candidate_member_to_ccs(row)
        for member_id in _flagged_members_from_doc(result_doc):
            ccs_id = member_to_ccs.get(member_id)
            if not ccs_id or (requested and ccs_id not in requested):
                continue
            entry = ccs_map.setdefault(ccs_id, {"members": set(), "draw_ids": set()})
            entry["members"].add(member_id)
            entry["draw_ids"].add(int(result_doc["draw_id"]))

    scores = [
        CcsScore(
            ccs_id=ccs_id,
            risk_tier="HIGH" if values["members"] else "LOW",
            flagged_member_count=len(values["members"]),
            flagged_members=sorted(values["members"]),
            evidence_draw_ids=sorted(values["draw_ids"]),
        )
        for ccs_id, values in ccs_map.items()
    ]
    scores.sort(key=lambda item: (item.flagged_member_count, len(item.evidence_draw_ids), item.ccs_id), reverse=True)
    return CcsScoreResponse(
        lookback_days=request.lookback_days,
        draws_scanned=len(candidate_rows),
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
    context: LiveScoringContext = Depends(get_live_scoring_context),
) -> AlertDrawResponse:
    lookback_days = max(1, min(int(lookback_days), 30))
    max_draws = max(1, min(int(max_draws), 50_000))
    limit = max(1, min(int(limit), 1_000))
    candidate_rows = _candidate_rows_for_window(
        context=context,
        lookback_days=lookback_days,
        max_draws=max_draws,
    )
    scorer = _make_scorer(context)
    alerts: list[AlertDraw] = []
    for result_doc, row in _candidate_result_docs(candidate_rows, scorer, context=context):
        flagged_members = sorted(_flagged_members_from_doc(result_doc))
        if not result_doc.get("requires_review") or not flagged_members:
            continue
        member_to_ccs = _candidate_member_to_ccs(row)
        alerts.append(
            AlertDraw(
                draw_id=int(result_doc["draw_id"]),
                draw_date=_candidate_draw_date(row),
                risk_tier="HIGH",
                partnership_count=len(result_doc.get("partnerships", [])),
                flagged_member_count=len(flagged_members),
                flagged_member_ids=flagged_members,
                ccs_ids=sorted({member_to_ccs.get(member, "UNKNOWN") for member in flagged_members}),
                max_stage1_score=float(result_doc.get("max_stage1_score") or 0.0),
                max_stage2_score=float(result_doc.get("max_stage2_score") or 0.0),
                response_details=list(result_doc.get("response_details", [])),
            )
        )
    alerts.sort(key=lambda item: (item.draw_date or "", item.draw_id), reverse=True)
    return AlertDrawResponse(
        lookback_days=lookback_days,
        draws_scanned=len(candidate_rows),
        alert_draw_count=len(alerts),
        alerts=alerts[:limit],
    )
