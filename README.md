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
                              # model training, evaluation, pusher, monitoring
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
  run_training.py             # full train → evaluate → monitor → promote
  run_batch_scoring.py        # score fresh cohort from promoted bundle
  audit_artifacts.py          # verify artifacts/current/ is consistent
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
    hybrid_scored_players.parquet
    alert_queue.csv
    hybrid_evaluation.json
    promotion_metadata.json

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
| `audit` | `python scripts/audit_artifacts.py` |
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
