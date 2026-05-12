from __future__ import annotations

from typing import Any

import pandas as pd


def latest_analyst_decisions(labels: list[dict[str, Any]] | pd.DataFrame) -> dict[tuple[int, str], str]:
    frame = pd.DataFrame(labels)
    if frame.empty or not {"draw_id", "member_id", "label"}.issubset(frame.columns):
        return {}
    frame = frame.loc[frame["label"].isin(["fraud", "not_fraud"])].copy()
    frame["draw_id"] = pd.to_numeric(frame["draw_id"], errors="coerce").astype("Int64")
    frame["member_id"] = frame["member_id"].astype(str).str.strip().str.upper()
    frame = frame.dropna(subset=["draw_id"])
    if "decided_at" in frame.columns:
        frame["decided_at"] = pd.to_datetime(frame["decided_at"], errors="coerce", utc=True)
        frame = frame.sort_values("decided_at")
    frame = frame.drop_duplicates(["draw_id", "member_id"], keep="last")
    return {
        (int(row.draw_id), str(row.member_id)): str(row.label)
        for row in frame[["draw_id", "member_id", "label"]].itertuples(index=False)
    }


def apply_pair_label_overrides(
    pair_df: pd.DataFrame,
    labels: list[dict[str, Any]] | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = pair_df.copy()
    if out.empty:
        return out, {"positive_overrides": 0, "negative_overrides": 0, "conflicts": 0}
    decisions = latest_analyst_decisions(labels)
    if not decisions:
        return out, {"positive_overrides": 0, "negative_overrides": 0, "conflicts": 0}

    for column, default in [
        ("label_stage1", 0),
        ("label_gold", 0),
        ("label_source", "strict_rule"),
        ("sample_weight", 1.0),
    ]:
        if column not in out.columns:
            out[column] = default

    positive = 0
    negative = 0
    conflicts = 0
    draw_ids = pd.to_numeric(out["draw_id"], errors="coerce").astype("Int64")
    for idx, draw_id in zip(out.index, draw_ids):
        if pd.isna(draw_id):
            continue
        member_a = str(out.at[idx, "member_a"]).strip().upper()
        member_b = str(out.at[idx, "member_b"]).strip().upper()
        decision_a = decisions.get((int(draw_id), member_a))
        decision_b = decisions.get((int(draw_id), member_b))
        if decision_a is None or decision_b is None:
            continue
        if decision_a == decision_b == "fraud":
            out.at[idx, "label_stage1"] = 1
            out.at[idx, "label_gold"] = 1
            out.at[idx, "label_source"] = "derived_pair_analyst"
            out.at[idx, "sample_weight"] = 1.0
            positive += 1
        elif decision_a == decision_b == "not_fraud":
            out.at[idx, "label_stage1"] = 0
            out.at[idx, "label_gold"] = 0
            out.at[idx, "label_source"] = "analyst_not_fraud_pair"
            out.at[idx, "sample_weight"] = 1.0
            negative += 1
        else:
            conflicts += 1
    return out, {"positive_overrides": positive, "negative_overrides": negative, "conflicts": conflicts}
