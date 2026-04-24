from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraud_detection.components.feature_engineering import (
    _aggregate_player_features_from_history,
    _normalize_raw_df,
    build_ccs_stats_lookup,
    compute_single_player_features,
)


def _raw_df() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for member_id, ccs_id, offset in [("A001", "CCS1", 0), ("A002", "CCS1", 10), ("B001", "CCS2", 20)]:
        for draw_idx in range(4):
            rows.append(
                {
                    "member_id": member_id,
                    "draw_id": offset + draw_idx,
                    "bets": json.dumps([{"number": str(draw_idx), "bet_amount": 5.0 + draw_idx}]),
                    "win_points": 2.0 + draw_idx,
                    "total_bet_amount": 5.0 + draw_idx,
                    "session_id": draw_idx // 2,
                    "ccs_id": ccs_id,
                    "createdAt": base + pd.Timedelta(days=offset + draw_idx),
                    "updatedAt": base + pd.Timedelta(days=offset + draw_idx),
                    "trans_date": base + pd.Timedelta(days=offset + draw_idx),
                }
            )
    return pd.DataFrame(rows)


def test_single_player_features_match_bulk_for_same_member():
    raw_df = _raw_df()
    normalized = _normalize_raw_df(raw_df)
    ccs_lookup = build_ccs_stats_lookup(normalized)

    bulk = _aggregate_player_features_from_history(normalized)
    bulk_row = bulk.loc[bulk["member_id"] == "A001"].reset_index(drop=True)

    single = compute_single_player_features(
        raw_df.loc[raw_df["member_id"] == "A001"].reset_index(drop=True),
        ccs_stats_lookup=ccs_lookup,
    ).reset_index(drop=True)

    assert list(single.columns) == list(bulk_row.columns)
    assert single.loc[0, "primary_ccs_id"] == bulk_row.loc[0, "primary_ccs_id"]

    for column in single.columns:
        left = single.loc[0, column]
        right = bulk_row.loc[0, column]
        if pd.api.types.is_number(left) and pd.api.types.is_number(right):
            np.testing.assert_allclose([left], [right], rtol=1e-9, atol=1e-12)
        else:
            assert left == right
