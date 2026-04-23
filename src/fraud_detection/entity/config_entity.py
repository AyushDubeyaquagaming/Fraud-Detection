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
    # Strategy-driven bounded ingestion (additive — all have safe defaults)
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
    output_dir: Path
    # Phase 2 rebaseline — primary gate is lift + capture_rate at top 5%.
    min_capture_rate_top_5pct: float = 0.40
    min_lift_top_5pct: float = 5.0
    # Fixed-count thresholds (top 50/500) reported for operational sizing.
    threshold_fixed_counts: list[int] = field(default_factory=lambda: [50, 500])
    # Alert queue size for the per-run alert_queue.csv output.
    alert_queue_size: int = 50
    # Retained for diagnostic reporting only — NOT used as gate.
    min_capture_top_20pct: int = 0


@dataclass
class ModelPusherConfig:
    current_dir: Path
    manifest_file: str = "serving_manifest.json"
    model_version: str = "hybrid_v1"
    # Phase 2 rebaseline — primary gate is lift + capture_rate at top 5%.
    min_capture_rate_top_5pct: float = 0.40
    min_lift_top_5pct: float = 5.0
    # Retained for diagnostic reporting only — NOT used as gate.
    min_capture_top_20pct: int = 0


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
