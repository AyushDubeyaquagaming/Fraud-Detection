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
    *,
    any_member_clique_override: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = pair_df.copy()
    if out.empty:
        return out, {"positive_overrides": 0, "negative_overrides": 0, "clique_positive_overrides": 0, "conflicts": 0}
    decisions = latest_analyst_decisions(labels)
    if not decisions:
        return out, {"positive_overrides": 0, "negative_overrides": 0, "clique_positive_overrides": 0, "conflicts": 0}

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
    clique_positive = 0
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
    if any_member_clique_override:
        clique_positive = _apply_any_member_clique_override(out, decisions)
    return out, {
        "positive_overrides": positive,
        "negative_overrides": negative,
        "clique_positive_overrides": clique_positive,
        "conflicts": conflicts,
    }


def _apply_any_member_clique_override(out: pd.DataFrame, decisions: dict[tuple[int, str], str]) -> int:
    strict_col = "is_strict_collusion_pattern" if "is_strict_collusion_pattern" in out.columns else "is_strict_match"
    strict_mask = pd.to_numeric(out.get(strict_col), errors="coerce").fillna(0).astype(int).eq(1)
    strict_pairs = out.loc[strict_mask].copy()
    if strict_pairs.empty:
        return 0
    override_count = 0
    for draw_id, group in strict_pairs.groupby(pd.to_numeric(strict_pairs["draw_id"], errors="coerce").astype("Int64")):
        if pd.isna(draw_id):
            continue
        components = _strict_pair_components(group)
        for members, row_indices in components:
            labels = [decisions.get((int(draw_id), member)) for member in members]
            if "fraud" not in labels or "not_fraud" in labels:
                continue
            for idx in row_indices:
                if int(pd.to_numeric(pd.Series([out.at[idx, "label_gold"]]), errors="coerce").fillna(0).iloc[0]) == 1:
                    continue
                out.at[idx, "label_stage1"] = 1
                out.at[idx, "label_gold"] = 1
                out.at[idx, "label_source"] = "derived_clique_analyst"
                out.at[idx, "sample_weight"] = 1.0
                override_count += 1
    return override_count


def _strict_pair_components(group: pd.DataFrame) -> list[tuple[set[str], list[Any]]]:
    adjacency: dict[str, set[str]] = {}
    member_rows: dict[str, list[Any]] = {}
    for idx, row in group.iterrows():
        a = str(row.get("member_a", "")).strip().upper()
        b = str(row.get("member_b", "")).strip().upper()
        if not a or not b:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        member_rows.setdefault(a, []).append(idx)
        member_rows.setdefault(b, []).append(idx)
    components = []
    visited: set[str] = set()
    for member in adjacency:
        if member in visited:
            continue
        stack = [member]
        members: set[str] = set()
        row_indices: set[Any] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            members.add(current)
            row_indices.update(member_rows.get(current, []))
            stack.extend(adjacency.get(current, set()) - visited)
        components.append((members, list(row_indices)))
    return components
