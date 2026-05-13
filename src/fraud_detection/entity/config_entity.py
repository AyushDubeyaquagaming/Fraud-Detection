from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataIngestionConfig:
    source: str
    parquet_path: Path
    mongo_uri_env_var: str
    mongo_database_env_var: str
    mongo_collection_env_var: str
    output_dir: Path
    parquet_strategy: str = "full_copy"
    parquet_strategy_params: dict[str, Any] = field(default_factory=dict)
    mongo_strategy: str = "date_window"
    mongo_strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataValidationConfig:
    schema_path: Path
    required_columns: list[str]
    min_row_count: int
    fraud_csv_path: Path
    output_dir: Path


@dataclass
class FeatureEngineeringConfig:
    fraud_csv_path: Path
    output_dir: Path
    mode: str = "training_eval"
    partnership: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTrainingConfig:
    random_seed: int
    output_dir: Path
    partnership: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluationConfig:
    output_dir: Path
    min_capture_rate_top_5pct: float = 0.40
    min_lift_top_5pct: float = 5.0


@dataclass
class ModelPusherConfig:
    current_dir: Path
    manifest_file: str = "serving_manifest.json"
    model_version: str = "partnership_v1"
    min_capture_rate_top_5pct: float = 0.40
    min_lift_top_5pct: float = 5.0
    register_on_promotion: bool = True
    registered_model_name: str = "fraud_detection_partnership_v1"
    archive_existing_staging: bool = True
    auto_promote_to_production: bool = False


@dataclass
class MonitoringConfig:
    enabled: bool
    reports_dir: str
    sample_size: int
    monitored_features: list[str]
    drift_threshold: float
    reference_from_current_metadata: bool


@dataclass
class PipelineConfig:
    artifact_root: Path
    run_id: str
    run_dir: Path
    current_dir: Path
    random_seed: int
    data_ingestion: DataIngestionConfig
    data_validation: DataValidationConfig
    feature_engineering: FeatureEngineeringConfig
    model_training: ModelTrainingConfig
    model_evaluation: ModelEvaluationConfig
    model_pusher: ModelPusherConfig
