from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.exception import FraudDetectionException
from fraud_detection.utils.mongodb import build_query_batches_from_strategy
from fraud_detection.utils.rolling_parquet_store import (
    materialize_rolling_window_to_parquet,
    write_dataframe_to_rolling_store,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "member_id": "A",
                "draw_id": 1,
                "bets": '[{"number": "1", "bet_amount": 1}]',
                "win_points": 0.0,
                "total_bet_amount": 1.0,
                "session_id": 1,
                "ccs_id": "CCS1",
                "createdAt": pd.Timestamp("2026-04-01T00:00:00Z"),
                "updatedAt": pd.Timestamp("2026-04-01T00:00:00Z"),
                "trans_date": pd.Timestamp("2026-04-01T00:00:00Z"),
            },
            {
                "member_id": "B",
                "draw_id": 2,
                "bets": '[{"number": "2", "bet_amount": 1}]',
                "win_points": 0.0,
                "total_bet_amount": 1.0,
                "session_id": 2,
                "ccs_id": "CCS2",
                "createdAt": pd.Timestamp("2026-04-02T00:00:00Z"),
                "updatedAt": pd.Timestamp("2026-04-02T00:00:00Z"),
                "trans_date": pd.Timestamp("2026-04-02T00:00:00Z"),
            },
        ]
    )


def test_write_and_materialize_rolling_window(tmp_path):
    store = tmp_path / "store"
    write_dataframe_to_rolling_store(_rows(), store)

    assert (store / "trans_date=2026-04-01").exists()
    assert (store / "trans_date=2026-04-02").exists()
    assert (store / "_schema.json").exists()

    output = tmp_path / "raw.parquet"
    stats = materialize_rolling_window_to_parquet(
        store,
        output,
        start_date="2026-04-02T00:00:00Z",
        end_date="2026-04-03T00:00:00Z",
    )
    df = pd.read_parquet(output)

    assert stats["row_count"] == 1
    assert stats["member_count"] == 1
    assert df["member_id"].tolist() == ["B"]


def test_rolling_store_strategy_is_not_a_mongo_query_builder():
    with pytest.raises(FraudDetectionException):
        build_query_batches_from_strategy("rolling_store", {})
