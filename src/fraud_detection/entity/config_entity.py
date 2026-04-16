from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataIngestionConfig:
    source: str  # "parquet" or "mongodb"
    parquet_path: Path
    mongo_uri_env_var: str
    mongo_database_env_var: str
    mongo_collection_env_var: str
    output_dir: Path


@dataclass
class DataValidationConfig:
    schema_path: Path
    required_columns: list[str]
    min_row_count: int
    fraud_csv_path: Path
    output_dir: Path


@dataclass
class FeatureEngineeringConfig:
    exclude_cols: list[str]
    log1p_cols: list[str]
    apply_pre_fraud_cutoff: bool
    fraud_csv_path: Path
    output_dir: Path
    mode: str = "training_eval"  # "training_eval" or "operational"


@dataclass
class ModelTrainingConfig:
    iso_forest_params: dict[str, Any]
    kmeans_params: dict[str, Any]
    lr_params: dict[str, Any]
    anomaly_weight: float
    supervised_weight: float
    random_seed: int
    output_dir: Path


@dataclass
class ModelEvaluationConfig:
    threshold_percentiles: list[float]
    risk_tier_p80: float
    risk_tier_p95: float
    min_capture_top_20pct: int
    output_dir: Path


@dataclass
class ModelPusherConfig:
    current_dir: Path
    min_capture_top_20pct: int


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
