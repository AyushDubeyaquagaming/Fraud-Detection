# BetBlitz Fraud Detection — MLOps Pipeline

Roulette fraud detection system using a hybrid unsupervised + supervised approach.
Source of truth: `notebook/03_hybrid_detection.ipynb`.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start local MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Run full training pipeline
python scripts/run_training.py

# Run batch scoring (requires promoted model bundle)
python scripts/run_batch_scoring.py

# Audit artifacts/current/
python scripts/audit_artifacts.py

# Run tests
pytest tests/
```

## Project Structure

```
src/fraud_detection/          # installable package
  components/                 # data ingestion, validation, feature engineering,
                              # model training, evaluation, pusher
  pipeline/                   # training_pipeline.py, batch_scoring_pipeline.py
  entity/                     # config_entity.py, artifact_entity.py
  utils/                      # common.py, mlflow_utils.py, mongodb.py
  constants/                  # constants.py
  logger.py / exception.py

configs/
  config.yaml                 # main pipeline config (edit source/paths here)
  model_params.yaml           # model hyperparameters (locked — do not change)
  schema.yaml                 # data schema

scripts/
  run_training.py             # full train → evaluate → promote
  run_batch_scoring.py        # score fresh cohort from promoted bundle
  audit_artifacts.py          # verify artifacts/current/ is consistent

artifacts/
  runs/run_YYYYMMDD_HHMMSS/   # per-run outputs (gitignored)
  current/                    # promoted production bundle (gitignored)
    model_bundle.joblib
    hybrid_scored_players.parquet
    alert_queue.csv
    hybrid_evaluation.json
    promotion_metadata.json

tests/
  unit/                       # fast unit tests
  integration/                # end-to-end pipeline test (requires data_cache/)
```

## Configuration

All tunable parameters live in `configs/`. Secrets go in `.env` (see `.env.example`).

The default training source is now live MongoDB. Every training run refreshes `data_cache/fraud_modeling_pull.parquet`
from MongoDB before validation and downstream processing so the cached parquet stays current.

| File | Purpose |
|---|---|
| `configs/config.yaml` | Pipeline settings, live data source, MLflow |
| `configs/model_params.yaml` | **Locked** model hyperparameters |
| `.env` | MongoDB URI, MLflow tracking URI |

## Model Parameters (locked per spec)

- IsolationForest: n_estimators=300, contamination=0.05, random_state=42
- KMeans: n_clusters=4, n_init=10, random_state=42
- LogisticRegression: C=0.1, class_weight="balanced", max_iter=2000
- Anomaly weight=0.60, Supervised weight=0.40

## Outputs

After a successful `run_training.py`:

- `artifacts/current/model_bundle.joblib` — all models + scalers + metadata
- `artifacts/current/hybrid_scored_players.parquet` — scored cohort
- `artifacts/current/alert_queue.csv` — top 50 players by risk score
- `artifacts/current/hybrid_evaluation.json` — capture rates, tier distribution
- `artifacts/runs/run_*/model_evaluation/plots/feature_importance.png` — supervised model importance
- `artifacts/runs/run_*/model_evaluation/plots/confusion_matrix.png` — out-of-sample confusion matrix
- `artifacts/runs/run_*/model_evaluation/plots/correlation_heatmap.png` — top-feature correlation map

These plots are also logged as MLflow artifacts for each run.

## Streamlit Demo

```bash
streamlit run streamlit_hybrid_demo.py
```

The demo automatically reads from `artifacts/current/` if present,
falling back to `data_cache/` for legacy compatibility.

Use `Internal validation mode` when replay-eval artifacts are available. With purely operational artifacts,
the demo still loads and shows behaviour space plus peer lookup, but label-only fields are hidden.

## Cohort Scope

> Scores are relative to the analysis cohort (~1,045 players), not the full BetBlitz platform.

## What's NOT in v1

FastAPI service, S3, Docker, Airflow, DVC, CI/CD — all deferred until pipeline is stable.
