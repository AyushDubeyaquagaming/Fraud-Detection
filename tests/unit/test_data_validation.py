from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.components.data_validation import DataValidation


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member_id": ["A001", "A002", "A003"] * 400,
            "draw_id": list(range(1200)),
            "total_bet_amount": [10.0] * 1200,
            "win_points": [5.0] * 1200,
            "bets": ['[{"number": "1", "bet_amount": 10}]'] * 1200,
            "createdAt": pd.date_range("2024-01-01", periods=1200, freq="1min"),
        }
    )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _make_fraud_csv(path: Path) -> None:
    pd.DataFrame({"member_id": ["A001"], "draw_id": [1]}).to_csv(path, index=False)


def _make_ingestion_artifact(raw_path: Path) -> DataIngestionArtifact:
    return DataIngestionArtifact(
        raw_data_path=raw_path,
        ingestion_report_path=raw_path.parent / "ingestion_report.json",
        row_count=1200,
        member_count=3,
        source_type="parquet",
    )


def _make_config(output_dir: Path, fraud_csv_path: Path, required_columns=None, min_row_count=1000):
    return DataValidationConfig(
        schema_path=Path("configs/schema.yaml"),
        required_columns=required_columns or ["member_id", "draw_id", "total_bet_amount", "win_points", "bets"],
        min_row_count=min_row_count,
        fraud_csv_path=fraud_csv_path,
        output_dir=output_dir,
    )


def test_valid_data_passes(tmp_path):
    df = _make_valid_df()
    raw_path = tmp_path / "raw_data.parquet"
    _write_parquet(df, raw_path)
    fraud_csv = tmp_path / "fraud.csv"
    _make_fraud_csv(fraud_csv)

    config = _make_config(tmp_path / "validation", fraud_csv)
    artifact = DataValidation(config, _make_ingestion_artifact(raw_path)).initiate_data_validation()

    assert artifact.is_valid is True
    assert artifact.validation_report_path.exists()


def test_missing_required_column_raises(tmp_path):
    df = _make_valid_df().drop(columns=["member_id"])
    raw_path = tmp_path / "raw_data.parquet"
    _write_parquet(df, raw_path)
    fraud_csv = tmp_path / "fraud.csv"
    _make_fraud_csv(fraud_csv)

    config = _make_config(tmp_path / "validation", fraud_csv)
    with pytest.raises(FraudDetectionException):
        DataValidation(config, _make_ingestion_artifact(raw_path)).initiate_data_validation()


def test_empty_dataframe_raises(tmp_path):
    df = _make_valid_df().iloc[:0]  # 0 rows
    raw_path = tmp_path / "raw_data.parquet"
    _write_parquet(df, raw_path)
    fraud_csv = tmp_path / "fraud.csv"
    _make_fraud_csv(fraud_csv)

    config = _make_config(tmp_path / "validation", fraud_csv, min_row_count=1000)
    with pytest.raises(FraudDetectionException):
        DataValidation(config, _make_ingestion_artifact(raw_path)).initiate_data_validation()
