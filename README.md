# BetBlitz Fraud Detection — MLOps Pipeline

Roulette partnership collusion detection system built around the candidate-draw store, Stage 1 pair scoring, and Stage 2 member scoring.

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

# Run tests
pytest tests/
```

## Project Structure

```
src/fraud_detection/          # installable package
  components/                 # data ingestion, validation, feature engineering,
                              # model training, evaluation, pusher, monitoring
  pipeline/                   # training_pipeline.py, batch_scoring_pipeline.py
  entity/                     # config_entity.py, artifact_entity.py
  utils/                      # common.py, mlflow_utils.py, mongodb.py
  constants/                  # constants.py
  logger.py / exception.py

configs/
  config.yaml                 # main pipeline config (edit source/paths here)
  schema.yaml                 # data schema
  config.local_training.yaml  # local candidate-store training config (gitignored)

scripts/
  run_training.py             # full train → evaluate → monitor → promote
  run_batch_scoring.py        # score fresh cohort from promoted bundle
  cleanup_old_runs.py         # artifact retention utility

orchestration/
  flows/training_flow.py      # Prefect-compatible training wrapper
  flows/batch_scoring_flow.py # Prefect-compatible scoring wrapper
  notifications.py            # Optional Slack alerts
  prefect.yaml                # Deployment spec for Prefect Cloud

artifacts/
  runs/run_YYYYMMDD_HHMMSS/   # per-run outputs (gitignored)
    monitoring/               # Evidently drift reports + drift_summary.json
  current/                    # promoted production bundle (gitignored)
    model_bundle.joblib
    stage1_model.joblib
    stage2_model.joblib
    partnership_table.parquet
    stage2_holdout_predictions.parquet
    live_predictions_backfill.parquet
    batch_scoring_report.json
    promotion_metadata.json
    serving_manifest.json

tests/
  unit/                       # fast unit tests (includes test_monitoring.py)
  integration/                # end-to-end pipeline test (requires data_cache/)
```

## Configuration

All tunable parameters live in `configs/`. Secrets go in `.env` (see `.env.example`).

The default training source is live MongoDB. A standard training run pulls a bounded cohort from MongoDB,
writes the raw pull into the run's ingestion artifact directory, validates it, and then continues through the
rest of the pipeline. The parquet path remains available for controlled replays and local debugging.

| File | Purpose |
|---|---|
| `configs/config.yaml` | Pipeline settings, live data source, MLflow |
| `configs/config.local_training.yaml` | Local candidate-store training window and thresholds |
| `.env` | MongoDB URI, MLflow tracking URI |

## Model Parameters (locked per spec)

- IsolationForest: n_estimators=300, contamination=0.05, random_state=42
- KMeans: n_clusters=4, n_init=10, random_state=42
- LogisticRegression: C=0.1, class_weight="balanced", max_iter=2000
- Anomaly weight=0.60, Supervised weight=0.40

## Outputs

After a successful `run_training.py`:

- `artifacts/current/model_bundle.joblib` — all models + scalers + metadata
- `artifacts/current/stage1_model.joblib` and `artifacts/current/stage2_model.joblib` — promoted serving models
- `artifacts/current/partnership_table.parquet` — promoted partnership context for serving
- `artifacts/current/stage2_holdout_predictions.parquet` — evaluation holdout predictions from the promoted run
- `artifacts/current/serving_manifest.json` and `artifacts/current/promotion_metadata.json` — current serving pointers and promotion metadata

After a successful `run_batch_scoring.py`:

- `artifacts/current/live_predictions_backfill.parquet` — batch-scored draw backfill
- `artifacts/current/batch_scoring_report.json` — scored draw counts and batch metadata

These plots are also logged as MLflow artifacts for each run.

## Streamlit Demo

```bash
streamlit run streamlit_partnership_demo.py
```

The Streamlit app calls the FastAPI live-scoring endpoints and reads recent predictions from MongoDB.
Run the API first, then launch the Streamlit app.

## Cohort Scope

> Scores are relative to the analysis cohort (~1,045 players), not the full BetBlitz platform.

## Monitoring

After each training run, Evidently drift reports are written to `artifacts/runs/<run_id>/monitoring/`:

| File | What it shows |
|---|---|
| `data_drift.html` | Raw data drift vs previous promoted run |
| `feature_drift.html` | Feature drift for 8 key signal columns |
| `prediction_drift.html` | Score distribution drift |
| `drift_summary.json` | Machine-readable summary with threshold status |

Reports are advisory only — they never block promotion. The first run after a fresh clone
will log a skip (no reference run exists yet). After a successful promotion and a second run,
full reports are generated.

## Docker

### Build

```powershell
docker build -t fraud-detection:local .
```

### Run training

```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/artifacts:/app/artifacts" `
  -v "${PWD}/logs:/app/logs" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -v "${PWD}/data_cache:/app/data_cache" `
  -v "${PWD}/configs:/app/configs:ro" `
  fraud-detection:local train
```

### Run batch scoring

```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/artifacts:/app/artifacts" `
  -v "${PWD}/logs:/app/logs" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -v "${PWD}/configs:/app/configs:ro" `
  fraud-detection:local score
```

### Start MLflow UI (local tracking only)

```powershell
docker compose up -d mlflow-ui
# Open http://localhost:5000
```

### Available container commands

| Command | What it runs |
|---|---|
| `train` | `python scripts/run_training.py` |
| `score` | `python scripts/run_batch_scoring.py` |
| `test` | `pytest tests/` |
| `shell` | Interactive shell in the container |
| `worker` | Prefect worker (Phase 3) |

## Artifact Retention

Remove old run directories while protecting `artifacts/current/`:

```powershell
# Dry run — see what would be deleted
python scripts/cleanup_old_runs.py --keep 5

# Actually delete
python scripts/cleanup_old_runs.py --keep 5 --execute
```

## Orchestration (Optional — Phase 3)

The training and scoring flows work as plain Python scripts with or without Prefect installed:

```powershell
python orchestration/flows/training_flow.py
python orchestration/flows/batch_scoring_flow.py
```

To use Prefect Cloud:

```powershell
pip install "prefect>=2.20.0,<3.0.0"
prefect cloud login
prefect work-pool create fraud-pool --type process
prefect deploy --prefect-file orchestration/prefect.yaml
prefect worker start --pool fraud-pool
```

Set `SLACK_WEBHOOK_URL` in `.env` to receive failure notifications.

See `DEPLOYMENT.md` for the full handoff guide.

## What's NOT in this version

FastAPI service, Kubernetes, Prometheus/Grafana, CI/CD, DVC, self-hosted Prefect server.
