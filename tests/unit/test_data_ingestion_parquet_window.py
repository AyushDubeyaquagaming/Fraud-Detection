from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.entity.config_entity import DataIngestionConfig


def test_parquet_date_window_stages_recent_rows(tmp_path: Path):
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "draw_id": [1, 2, 3],
            "bets": ["[]", "[]", "[]"],
            "win_points": [0.0, 0.0, 0.0],
            "total_bet_amount": [10.0, 20.0, 30.0],
            "trans_date": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-05T00:00:00Z",
                    "2026-04-10T00:00:00Z",
                ],
                utc=True,
            ),
        }
    ).to_parquet(source_path, index=False)

    config = DataIngestionConfig(
        source="parquet",
        parquet_path=source_path,
        mongo_uri_env_var="MONGODB_URI",
        mongo_database_env_var="MONGODB_DATABASE",
        mongo_collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        output_dir=tmp_path / "out",
        parquet_strategy="date_window",
        parquet_strategy_params={
            "timestamp_field": "trans_date",
            "lookback_days": 5,
        },
    )

    artifact = DataIngestion(config).initiate_data_ingestion()
    staged = pd.read_parquet(artifact.raw_data_path)

    assert staged["draw_id"].tolist() == [2, 3]
    assert artifact.row_count == 2
    assert artifact.member_count == 2