"""Integration tests for bounded MongoDB ingestion.

These tests hit a real MongoDB instance and are skipped by default.
Set the environment variable RUN_MONGO_TESTS=1 to enable them.

Usage:
    RUN_MONGO_TESTS=1 pytest tests/integration/test_mongodb_ingestion.py -v

The tests check correctness (row counts, metadata, artifact shape) not
elapsed-time thresholds — runtime is logged as an operational benchmark only.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Gate: skip the entire module unless RUN_MONGO_TESTS=1
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MONGO_TESTS", "0") != "1",
    reason="Set RUN_MONGO_TESTS=1 to run live MongoDB integration tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from fraud_detection.constants.constants import REPO_ROOT
from fraud_detection.utils.mongodb import (
    build_query_batches_from_strategy,
    pull_query_batches_to_dataframe,
    stream_query_batches_to_parquet,
)


def _recent_date_window_params(days: int = 30) -> dict:
    return {"timestamp_field": "trans_date", "lookback_days": days}


# ---------------------------------------------------------------------------
# date_window: live query returns rows
# ---------------------------------------------------------------------------


def test_date_window_returns_rows(tmp_path):
    """date_window strategy against a known recent range must return >0 rows."""
    params = _recent_date_window_params(days=30)
    queries = build_query_batches_from_strategy("date_window", params)

    start = time.perf_counter()
    df = pull_query_batches_to_dataframe(
        uri_env_var="MONGODB_URI",
        db_env_var="MONGODB_DATABASE",
        collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        query_filters=queries,
    )
    elapsed = time.perf_counter() - start

    assert len(df) > 0, "Expected >0 rows for a 30-day window"
    assert "member_id" in df.columns
    print(
        f"\ndate_window (30d): {len(df):,} rows, "
        f"{df['member_id'].nunique():,} members, elapsed={elapsed:.1f}s"
    )


def test_date_window_stream_to_parquet(tmp_path):
    """date_window streaming path must produce a valid parquet file with >0 rows."""
    params = _recent_date_window_params(days=30)
    queries = build_query_batches_from_strategy("date_window", params)
    output_path = tmp_path / "test_pull.parquet"

    start = time.perf_counter()
    stats = stream_query_batches_to_parquet(
        uri_env_var="MONGODB_URI",
        db_env_var="MONGODB_DATABASE",
        collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        output_paths=[output_path],
        query_filters=queries,
    )
    elapsed = time.perf_counter() - start

    assert output_path.exists(), "Expected parquet file to be created"
    assert stats["row_count"] > 0
    assert stats["member_count"] > 0
    assert stats["query_count"] == 1
    assert "date_range" in stats

    import pandas as pd
    df = pd.read_parquet(output_path)
    assert len(df) == stats["row_count"]
    print(
        f"\ndate_window stream (30d): {stats['row_count']:,} rows, "
        f"{stats['member_count']:,} members, elapsed={elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# member_list: fraud CSV members return rows
# ---------------------------------------------------------------------------


def test_member_list_from_fraud_csv_returns_rows(tmp_path):
    """member_list strategy using the fraud CSV must return >0 rows and valid output."""
    fraud_csv_path = REPO_ROOT / "ROULET CHEATING DATA.csv"
    if not fraud_csv_path.exists():
        pytest.skip(f"Fraud CSV not found: {fraud_csv_path}")

    import pandas as pd
    fraud_df = pd.read_csv(fraud_csv_path)
    if "member_id" not in fraud_df.columns:
        pytest.skip("Fraud CSV does not have a member_id column")
    if fraud_df["member_id"].dropna().empty:
        pytest.skip("Fraud CSV member_id column is empty")

    params = {"member_ids_source": "fraud_csv"}
    queries = build_query_batches_from_strategy(
        "member_list", params, fraud_csv_path=fraud_csv_path
    )

    assert len(queries) >= 1

    start = time.perf_counter()
    df = pull_query_batches_to_dataframe(
        uri_env_var="MONGODB_URI",
        db_env_var="MONGODB_DATABASE",
        collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        query_filters=queries,
    )
    elapsed = time.perf_counter() - start

    assert len(df) > 0, "Expected >0 rows for fraud CSV member list"
    print(
        f"\nmember_list (fraud_csv): {len(df):,} rows, "
        f"{df['member_id'].nunique():,} members, {len(queries)} query batch(es), "
        f"elapsed={elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# full_collection: requires confirmation
# ---------------------------------------------------------------------------


def test_full_collection_raises_without_confirm():
    """full_collection must raise before touching MongoDB if confirm_full_pull is false."""
    from fraud_detection.exception import FraudDetectionException
    params = {"confirm_full_pull": False}
    with pytest.raises(FraudDetectionException):
        build_query_batches_from_strategy("full_collection", params)


# ---------------------------------------------------------------------------
# Ingestion report metadata
# ---------------------------------------------------------------------------


def test_ingestion_report_includes_strategy_metadata(tmp_path):
    """DataIngestion artifact must include strategy_used and query_count."""
    from dataclasses import fields as dc_fields
    from fraud_detection.entity.config_entity import DataIngestionConfig
    from fraud_detection.components.data_ingestion import DataIngestion

    config = DataIngestionConfig(
        source="mongodb",
        parquet_path=tmp_path / "cache.parquet",
        mongo_uri_env_var="MONGODB_URI",
        mongo_database_env_var="MONGODB_DATABASE",
        mongo_collection_env_var="MONGODB_COLLECTION_ROULETTE_REPORT",
        output_dir=tmp_path / "ingestion",
        mongo_strategy="date_window",
        mongo_strategy_params={"timestamp_field": "trans_date", "lookback_days": 14},
    )

    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.strategy_used == "date_window"
    assert artifact.query_count == 1
    assert artifact.row_count > 0

    import json
    with open(artifact.ingestion_report_path) as f:
        report = json.load(f)
    assert report["strategy_used"] == "date_window"
    assert report["query_count"] == 1
    print(
        f"\ningestion report: {artifact.row_count:,} rows, "
        f"strategy={artifact.strategy_used}, query_count={artifact.query_count}"
    )
