from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    raw_data_path: Path
    ingestion_report_path: Path
    row_count: int
    member_count: int
    source_type: str
    # Strategy audit fields (additive — optional with safe defaults)
    strategy_used: str | None = None
    query_count: int = 1
    date_range: dict[str, str] | None = None


@dataclass
class DataValidationArtifact:
    validation_report_path: Path
    is_valid: bool
    message: str


@dataclass
class FeatureEngineeringArtifact:
    player_features_path: Path
    history_df_path: Path
    fraud_player_count: int
    dropped_positive_count: int
    feature_columns: list[str]
    feature_summary_path: Path
    mode: str


@dataclass
class ModelTrainingArtifact:
    iso_forest_path: Path
    kmeans_path: Path
    mahalanobis_stats_path: Path
    scaler_path: Path
    lr_operational_path: Path
    training_report_path: Path
    feature_columns: list[str]


@dataclass
class ModelEvaluationArtifact:
    scored_players_path: Path
    capture_rate_table_path: Path
    evaluation_report_path: Path
    combined_oos_top_20pct: int
    gate_passed: bool


@dataclass
class ModelPusherArtifact:
    model_bundle_path: Path
    promotion_metadata_path: Path
    promoted: bool
