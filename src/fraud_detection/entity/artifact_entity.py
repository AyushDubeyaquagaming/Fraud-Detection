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
    draw_features_path: Path | None = None
    stage1_features_path: Path | None = None
    stage1_labels_path: Path | None = None
    stage1_oof_predictions_path: Path | None = None
    stage2_features_path: Path | None = None
    partnership_table_path: Path | None = None
    pair_events_path: Path | None = None


@dataclass
class ModelTrainingArtifact:
    training_report_path: Path
    feature_columns: list[str]
    stage1_model_path: Path | None = None
    stage2_model_path: Path | None = None
    model_bundle_path: Path | None = None
    stage1_oof_predictions_path: Path | None = None
    stage2_features_path: Path | None = None
    partnership_table_path: Path | None = None
    stage1_feature_columns: list[str] | None = None
    stage2_feature_columns: list[str] | None = None


@dataclass
class ModelEvaluationArtifact:
    stage2_holdout_predictions_path: Path
    capture_rate_table_path: Path
    evaluation_report_path: Path
    gate_passed: bool
    stage2_capture_rate_top_5pct: float
    stage2_lift_top_5pct: float
    stage2_top_50_captured: int


@dataclass
class ModelPusherArtifact:
    model_bundle_path: Path
    promotion_metadata_path: Path
    promoted: bool
    # Optional registry coordinates — populated only when registration succeeds.
    # Registry is best-effort; absence here does not affect serving.
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    registered_model_stage: str | None = None


@dataclass
class MonitoringArtifact:
    reports_dir: Path | None
    data_drift_report_path: Path | None
    feature_drift_report_path: Path | None
    prediction_drift_report_path: Path | None
    drift_summary_path: Path | None
    monitoring_completed: bool
