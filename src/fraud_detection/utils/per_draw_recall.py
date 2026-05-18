from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.utils.fraud_label_matching import (
    MATCHED,
    classify_fraud_csv,
    summarize_verdicts,
)


def build_available_keys_from_candidate_store(
    candidate_store_path: Path,
    *,
    member_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not candidate_store_path.exists():
        return pd.DataFrame(columns=["member_id", "draw_id", "trans_date_min"])
    import pyarrow.dataset as ds

    dataset = ds.dataset(candidate_store_path, format="parquet", partitioning="hive")
    columns = [column for column in ["draw_id", "member_ids", "trans_date_min"] if column in dataset.schema.names]
    if not {"draw_id", "member_ids"}.issubset(columns):
        return pd.DataFrame(columns=["member_id", "draw_id", "trans_date_min"])
    target_members = {str(member_id).strip().upper() for member_id in member_ids or []}
    for batch in dataset.scanner(columns=columns, batch_size=100_000).to_batches():
        chunk = batch.to_pandas()
        for item in chunk.to_dict("records"):
            member_ids = item.get("member_ids")
            if member_ids is None:
                continue
            for member_id in list(member_ids):
                normalized = str(member_id).strip().upper()
                if target_members and normalized not in target_members:
                    continue
                rows.append(
                    {
                        "member_id": normalized,
                        "draw_id": item.get("draw_id"),
                        "trans_date_min": item.get("trans_date_min"),
                    }
                )
    return pd.DataFrame(rows)


def build_per_draw_recall_report(
    *,
    fraud_csv_path: Path,
    available_keys_df: pd.DataFrame,
    stage1_predictions: pd.DataFrame,
    stage2_predictions: pd.DataFrame | None,
    stage1_threshold: float,
    stage2_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    fraud_csv = pd.read_csv(fraud_csv_path)
    verdicts = classify_fraud_csv(fraud_csv, available_keys_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = _normalize_stage1(stage1_predictions)
    stage2 = _normalize_stage2(stage2_predictions)
    matched = verdicts.loc[verdicts["final_status"].eq(MATCHED)].copy()
    draw_rows = []
    for draw_id, draw_group in matched.groupby("fraud_csv_draw_id", dropna=True):
        members = set(draw_group["fraud_csv_member_id"].astype(str).str.upper())
        draw_stage1 = stage1.loc[stage1["draw_id"].eq(int(draw_id))]
        member_pair_mask = draw_stage1["member_a"].isin(members) | draw_stage1["member_b"].isin(members)
        stage1_hit = bool(
            (
                draw_stage1.loc[member_pair_mask, "stage1_score"]
                >= float(stage1_threshold)
            ).any()
        )
        stage2_hit = bool(stage2.loc[stage2["member_id"].isin(members), "stage2_score"].ge(float(stage2_threshold)).any())
        draw_rows.append(
            {
                "draw_id": int(draw_id),
                "fraud_member_count": int(len(members)),
                "stage1_pair_flagged": stage1_hit,
                "member_stage2_escalated_via_participation": stage2_hit,
                "caught_by_any_signal": bool(stage1_hit or stage2_hit),
            }
        )
    per_draw = pd.DataFrame(
        draw_rows,
        columns=[
            "draw_id",
            "fraud_member_count",
            "stage1_pair_flagged",
            "member_stage2_escalated_via_participation",
            "caught_by_any_signal",
        ],
    )
    per_draw_path = output_dir / "per_draw_recall_verdicts.parquet"
    per_draw.to_parquet(per_draw_path, index=False)
    verdict_path = output_dir / "fraud_label_coverage.parquet"
    verdicts.to_parquet(verdict_path, index=False)
    summary = {
        **summarize_verdicts(verdicts),
        "score_eligible_draws": int(len(per_draw)),
        "stage1_draws_caught": int(per_draw.get("stage1_pair_flagged", pd.Series(dtype=bool)).sum()),
        "stage2_draws_caught_via_participation": int(
            per_draw.get("member_stage2_escalated_via_participation", pd.Series(dtype=bool)).sum()
        ),
        "caught_by_any_signal": int(per_draw.get("caught_by_any_signal", pd.Series(dtype=bool)).sum()),
        "false_negative_draws": int((~per_draw.get("caught_by_any_signal", pd.Series(dtype=bool))).sum()) if not per_draw.empty else 0,
        "per_draw_verdicts_path": str(per_draw_path),
        "fraud_label_coverage_path": str(verdict_path),
        "stage2_note": "member_stage2_escalated_via_participation reuses member-level Stage 2 scores across draws; it is not a draw-specific Stage 2 score.",
    }
    per_draw_report_path = output_dir / "per_draw_recall_report.json"
    coverage_summary_path = output_dir / "fraud_label_coverage.json"
    per_draw_report_path.write_text(
        __import__("json").dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    coverage_summary_path.write_text(
        __import__("json").dumps(summarize_verdicts(verdicts), indent=2, default=str),
        encoding="utf-8",
    )
    return {
        **summary,
        "per_draw_recall_report_path": str(per_draw_report_path),
        "fraud_label_coverage_report_path": str(verdict_path),
        "fraud_label_coverage_summary_path": str(coverage_summary_path),
    }


def build_batch_false_negative_report(
    *,
    fraud_csv_path: Path,
    scored_predictions_path: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    output_path: Path,
) -> dict[str, Any]:
    fraud = pd.read_csv(fraud_csv_path)
    fraud.columns = [str(column).strip().lower() for column in fraud.columns]
    if "date" in fraud.columns:
        fraud["date_parsed"] = pd.to_datetime(fraud["date"], errors="coerce", utc=True)
        fraud = fraud.loc[fraud["date_parsed"].ge(window_start) & fraud["date_parsed"].lt(window_end)].copy()
    fraud["draw_id_norm"] = pd.to_numeric(fraud.get("draw_id"), errors="coerce").astype("Int64")
    scored = pd.read_parquet(scored_predictions_path) if scored_predictions_path.exists() else pd.DataFrame()
    if scored.empty:
        scored = pd.DataFrame(columns=["draw_id", "requires_review"])
    scored["draw_id_norm"] = pd.to_numeric(scored.get("draw_id"), errors="coerce").astype("Int64")
    scored["flagged"] = _scored_flagged(scored)
    scored_lookup = scored.dropna(subset=["draw_id_norm"]).drop_duplicates("draw_id_norm").set_index("draw_id_norm")["flagged"]
    rows = []
    for draw_id, group in fraud.dropna(subset=["draw_id_norm"]).groupby("draw_id_norm"):
        flagged = bool(scored_lookup.get(draw_id, False))
        rows.append(
            {
                "draw_id": int(draw_id),
                "fraud_csv_rows": int(len(group)),
                "scored_in_batch": bool(draw_id in scored_lookup.index),
                "flagged_in_batch": flagged,
                "false_negative": bool(not flagged),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["draw_id", "fraud_csv_rows", "scored_in_batch", "flagged_in_batch", "false_negative"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts_path = output_path.with_suffix(".parquet")
    frame.to_parquet(verdicts_path, index=False)
    report = {
        "batch_window_start": window_start.isoformat(),
        "batch_window_end": window_end.isoformat(),
        "fraud_draws_in_window": int(len(frame)),
        "fraud_draws_scored": int(frame.get("scored_in_batch", pd.Series(dtype=bool)).sum()),
        "fraud_draws_flagged": int(frame.get("flagged_in_batch", pd.Series(dtype=bool)).sum()),
        "false_negative_draws": int(frame.get("false_negative", pd.Series(dtype=bool)).sum()),
        "verdicts_path": str(verdicts_path),
    }
    output_path.write_text(__import__("json").dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _normalize_stage1(stage1_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = stage1_predictions.copy()
    for column in ["member_a", "member_b", "member_id"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip().str.upper()
    if "member_a" not in frame.columns and "member_id" in frame.columns:
        frame["member_a"] = frame["member_id"]
    if "member_b" not in frame.columns:
        frame["member_b"] = ""
    frame["draw_id"] = pd.to_numeric(frame.get("draw_id"), errors="coerce").astype("Int64")
    frame["stage1_score"] = pd.to_numeric(frame.get("stage1_score"), errors="coerce").fillna(0.0)
    return frame.dropna(subset=["draw_id"])


def _normalize_stage2(stage2_predictions: pd.DataFrame | None) -> pd.DataFrame:
    if stage2_predictions is None or stage2_predictions.empty:
        return pd.DataFrame(columns=["member_id", "stage2_score"])
    frame = stage2_predictions.copy()
    frame["member_id"] = frame.get("member_id", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    frame["stage2_score"] = pd.to_numeric(frame.get("stage2_score"), errors="coerce").fillna(0.0)
    return frame


def _scored_flagged(scored: pd.DataFrame) -> pd.Series:
    if "requires_review" in scored.columns:
        return scored["requires_review"].fillna(False).astype(bool)
    return pd.Series(False, index=scored.index)
