from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection.components.feature_engineering import (
    FeatureEngineering,
    gini_coeff,
    make_bet_template,
    safe_entropy,
)


# --- safe_entropy tests ---

def test_entropy_uniform():
    result = safe_entropy([1, 1, 1, 1])
    assert abs(result - math.log2(4)) < 1e-9


def test_entropy_concentrated():
    result = safe_entropy([10, 0, 0, 0])
    assert result == 0.0


def test_entropy_zeros():
    assert safe_entropy([0, 0, 0]) == 0.0


# --- gini_coeff tests ---

def test_gini_uniform():
    result = gini_coeff([5, 5, 5, 5])
    assert abs(result) < 1e-9


def test_gini_empty():
    assert gini_coeff([]) == 0.0


def test_gini_positive():
    result = gini_coeff([1, 10, 100])
    assert 0.0 <= result <= 1.0


# --- make_bet_template tests ---

def test_template_order_invariant():
    bets_a = [{"number": "5", "bet_amount": 10}, {"number": "3", "bet_amount": 20}]
    bets_b = [{"number": "3", "bet_amount": 20}, {"number": "5", "bet_amount": 10}]
    assert make_bet_template(bets_a) == make_bet_template(bets_b)


def test_template_excludes_zero_bets():
    bets = [{"number": "1", "bet_amount": 0}, {"number": "2", "bet_amount": 5}]
    template = make_bet_template(bets)
    assert len(template) == 1
    assert template[0][0] == "2"


def test_template_empty_list():
    assert make_bet_template([]) == tuple()


def test_template_non_list():
    assert make_bet_template(None) == tuple()


# --- Pre-fraud cutoff integration test ---

def test_pre_fraud_cutoff_keeps_only_pre_fraud_rows():
    """Player with fraud at draw 100 → only draws < 100 retained."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Build synthetic raw data: player A001 plays draws 50-150
        rows = []
        for draw in range(50, 151):
            rows.append({
                "member_id": "A001",
                "draw_id": draw,
                "total_bet_amount": 10.0,
                "win_points": 5.0,
                "bets": '[{"number": "1", "bet_amount": 10}]',
                "session_id": 1,
                "ccs_id": "CCS1",
                "createdAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw),
            })
        raw_df = pd.DataFrame(rows)
        raw_path = tmpdir / "raw_data.parquet"
        raw_df.to_parquet(raw_path, index=False)

        # Fraud CSV says A001 committed fraud at draw 100
        fraud_csv_path = tmpdir / "fraud.csv"
        pd.DataFrame({"member_id": ["A001"], "draw_id": [100]}).to_csv(fraud_csv_path, index=False)

        config = FeatureEngineeringConfig(
            exclude_cols=["member_id", "event_fraud_flag", "primary_ccs_id",
                          "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player"],
            log1p_cols=[],
            apply_pre_fraud_cutoff=True,
            fraud_csv_path=fraud_csv_path,
            output_dir=tmpdir / "fe_output",
            mode="training_eval",
        )
        ingestion_artifact = DataIngestionArtifact(
            raw_data_path=raw_path,
            ingestion_report_path=raw_path.parent / "ingestion_report.json",
            row_count=len(raw_df),
            member_count=1,
            source_type="parquet",
        )
        artifact = FeatureEngineering(config, ingestion_artifact).initiate_feature_engineering()

        # Check history_df only has pre-fraud draws
        history_df = pd.read_parquet(artifact.history_df_path)
        history_for_a001 = history_df[history_df["member_id"] == "A001"]

        # draw_id is Int64, cast for comparison
        draws = set(int(d) for d in history_for_a001["draw_id"].dropna())
        assert all(d < 100 for d in draws), f"Expected only draws < 100, got: {sorted(draws)}"


def test_fraud_at_draw_zero_dropped():
    """Player with fraud at draw 0 (first draw) → no pre-fraud history → dropped."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        rows = []
        for draw in range(0, 5):
            rows.append({
                "member_id": "B001",
                "draw_id": draw,
                "total_bet_amount": 10.0,
                "win_points": 5.0,
                "bets": '[{"number": "1", "bet_amount": 10}]',
                "session_id": 1,
                "ccs_id": "CCS2",
                "createdAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw),
            })
        raw_df = pd.DataFrame(rows)
        raw_path = tmpdir / "raw_data.parquet"
        raw_df.to_parquet(raw_path, index=False)

        fraud_csv_path = tmpdir / "fraud.csv"
        pd.DataFrame({"member_id": ["B001"], "draw_id": [0]}).to_csv(fraud_csv_path, index=False)

        config = FeatureEngineeringConfig(
            exclude_cols=["member_id", "event_fraud_flag", "primary_ccs_id",
                          "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player"],
            log1p_cols=[],
            apply_pre_fraud_cutoff=True,
            fraud_csv_path=fraud_csv_path,
            output_dir=tmpdir / "fe_output",
            mode="training_eval",
        )
        ingestion_artifact = DataIngestionArtifact(
            raw_data_path=raw_path,
            ingestion_report_path=raw_path.parent / "ingestion_report.json",
            row_count=len(raw_df),
            member_count=1,
            source_type="parquet",
        )
        artifact = FeatureEngineering(config, ingestion_artifact).initiate_feature_engineering()

        player_features = pd.read_parquet(artifact.player_features_path)
        # B001 had fraud at draw 0, no pre-fraud draws → should not appear in player features
        assert "B001" not in player_features["member_id"].values


def test_history_schema_stable_across_null_and_typed_fraud_columns(tmp_path):
    history_path = tmp_path / "history_df.parquet"
    writers = {}

    base_columns = {
        "member_id": ["A001"],
        "draw_id": pd.Series([1], dtype="Int64"),
        "bets": ['[{"number": "1", "bet_amount": 10}]'],
        "win_points": [5.0],
        "total_bet_amount": [10.0],
        "session_id": [1],
        "ccs_id": ["CCS1"],
        "createdAt": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "updatedAt": [pd.Timestamp("2024-01-01T00:00:30Z")],
        "trans_date": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "ts": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "bets_per_draw": [1],
        "nonzero_bets_per_draw": [1],
        "tiny_bet_ratio_in_draw": [0.0],
        "max_bet_share_in_draw": [1.0],
        "bet_amount_std_in_draw": [0.0],
        "bet_amount_mean_in_draw": [10.0],
        "entropy_in_draw": [0.0],
        "gini_in_draw": [0.0],
        "unique_positions_in_draw": [1],
        "position_coverage": [1 / 38.0],
        "net_result": [-5.0],
        "bet_template": ['[["1",10.0]]'],
        "fraud_event_key": ["1|A001"],
        "inter_draw_seconds": [0.0],
        "event_label": [0],
        "is_fraud_player": [0],
    }

    null_df = FeatureEngineering._ensure_history_schema(pd.DataFrame(base_columns))

    typed_df = pd.DataFrame(
        {
            **base_columns,
            "member_id": ["A002"],
            "draw_id": pd.Series([2], dtype="Int64"),
            "fraud_event_key": ["2|A002"],
            "event_label": [1],
            "first_fraud_ts": [pd.Timestamp("2024-01-01T00:01:00Z")],
            "first_fraud_draw_id": pd.Series([2], dtype="Int64"),
            "is_fraud_player": [1],
        }
    )
    typed_df = FeatureEngineering._ensure_history_schema(typed_df)

    FeatureEngineering._append_dataframe_to_parquet(null_df, history_path, writers)
    FeatureEngineering._append_dataframe_to_parquet(typed_df, history_path, writers)
    FeatureEngineering._close_writers(writers)

    history_df = pd.read_parquet(history_path)
    assert len(history_df) == 2
    assert str(history_df["first_fraud_ts"].dtype).startswith("datetime64")
    assert str(history_df["first_fraud_draw_id"].dtype) == "Int64"
