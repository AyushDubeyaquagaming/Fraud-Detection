from __future__ import annotations

from pathlib import Path

# --- Repository root ---
REPO_ROOT = Path(__file__).resolve().parents[3]

# --- Config paths ---
CONFIG_FILE_PATH = REPO_ROOT / "configs" / "config.yaml"
BATCH_SCORING_CONFIG_FILE_PATH = REPO_ROOT / "configs" / "batch_scoring.yaml"
SCHEMA_FILE_PATH = REPO_ROOT / "configs" / "schema.yaml"

# --- Artifact directories ---
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
RUNS_DIR = ARTIFACT_ROOT / "runs"
CURRENT_DIR = ARTIFACT_ROOT / "current"

# --- Data paths ---
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
FRAUD_CSV_PATH = REPO_ROOT / "ROULET CHEATING DATA.csv"

# --- Env var names ---
ENV_MONGODB_URI = "MONGODB_URI"
ENV_MONGODB_DATABASE = "MONGODB_DATABASE"
ENV_MONGODB_COLLECTION = "MONGODB_COLLECTION_ROULETTE_REPORT"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_MLFLOW_TRACKING_USERNAME = "MLFLOW_TRACKING_USERNAME"
ENV_MLFLOW_TRACKING_PASSWORD = "MLFLOW_TRACKING_PASSWORD"

# --- Artifact file names ---
RAW_DATA_FILE = "raw_data.parquet"
INGESTION_REPORT_FILE = "ingestion_report.json"
VALIDATION_REPORT_FILE = "validation_report.json"
HISTORY_DF_FILE = "history_df.parquet"
PLAYER_FEATURES_FILE = "player_features.parquet"
FEATURE_SUMMARY_FILE = "feature_summary.json"
STAGE1_MODEL_FILE = "stage1_model.joblib"
STAGE2_MODEL_FILE = "stage2_model.joblib"
TRAINING_REPORT_FILE = "training_report.json"
CAPTURE_RATE_TABLE_FILE = "capture_rate_table.csv"
SCORED_PLAYERS_FILE = "stage2_holdout_predictions.parquet"
EVALUATION_REPORT_FILE = "evaluation_report.json"
MODEL_BUNDLE_FILE = "model_bundle.joblib"
PARTNERSHIP_TABLE_FILE = "partnership_table.parquet"
PROMOTION_METADATA_FILE = "promotion_metadata.json"
RUN_METADATA_FILE = "run_metadata.json"
BATCH_SCORING_REPORT_FILE = "batch_scoring_report.json"

# --- Model registry ---
MLFLOW_EXPERIMENT_NAME = "fraud_detection_partnership_v1"
MLFLOW_REGISTERED_MODEL_NAME = "fraud_detection_partnership_v1"

# --- Random seed ---
RANDOM_SEED = 42
