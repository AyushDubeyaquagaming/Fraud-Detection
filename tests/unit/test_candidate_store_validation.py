from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_detection.components.data_validation import DataValidation
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.exception import FraudDetectionException


def _artifact(path):
    return DataIngestionArtifact(
        raw_data_path=path,
        ingestion_report_path=path / "ingestion_report.json",
        row_count=1,
        member_count=2,
        source_type="candidate_store",
    )


def _config(tmp_path):
    return DataValidationConfig(
        schema_path=tmp_path / "schema.yaml",
        required_columns=[],
        min_row_count=1,
        fraud_csv_path=tmp_path / "fraud.csv",
        output_dir=tmp_path / "validation",
    )


def test_candidate_store_validation_accepts_valid_schema(tmp_path) -> None:
    store = tmp_path / "candidate_draws" / "year=2026" / "month=04" / "week=18"
    store.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "draw_id": 1,
                "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "trans_date_max": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "qualifying_player_count": 2,
                "member_ids": ["A", "B"],
                "coverage_bytes": [b"\x01" * 38, b"\x00" * 38],
                "amount_vector": [[1.0] * 38, [0.0] * 38],
                "total_bet_amounts": [1000.0, 1000.0],
                "win_points": [0.0, 0.0],
            }
        ]
    )
    pq.write_table(table, store / "draws.parquet")

    artifact = DataValidation(_config(tmp_path), _artifact(tmp_path / "candidate_draws")).initiate_data_validation()

    assert artifact.is_valid


def test_candidate_store_validation_fails_on_missing_columns(tmp_path) -> None:
    store = tmp_path / "candidate_draws" / "year=2026" / "month=04" / "week=18"
    store.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"draw_id": 1}]), store / "draws.parquet")

    with pytest.raises(FraudDetectionException, match="candidate store validation failed"):
        DataValidation(_config(tmp_path), _artifact(tmp_path / "candidate_draws")).initiate_data_validation()
