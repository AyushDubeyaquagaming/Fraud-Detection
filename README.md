# BetBlitz Fraud Detection - MLOps Pipeline

Roulette partnership collusion detection system built around the candidate-draw store, Stage 1 pair scoring, Stage 2 member scoring, CCS profit context, and analyst feedback labels.

## Quick Start

```powershell
# Install dependencies
pip install -e .

# Start local MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Run the full candidate-store cycle
python scripts/run_full_cycle.py --config configs/config.yaml

# Run training only when upstream parquet stores already exist
python scripts/run_training.py --config configs/config.yaml

# Run batch scoring from promoted artifacts
python scripts/run_batch_scoring.py --config configs/batch_scoring.yaml

# Run tests
pytest tests/
```

## Operational Path

The supported production path is candidate-store based:

1. Extract candidate roulette draws from MongoDB into `data_store/candidate_draws`.
2. Refresh daily CCS-member profit into `data_store/ccs_daily_profit`.
3. Train from those parquet stores.
4. Promote the latest successfully trained model bundle.
5. Run batch scoring after promotion.

Analyst labels are optional feedback. Missing labels are logged as unavailable and do not block training or promotion.

## Project Structure

```text
src/fraud_detection/
  components/                 # feature engineering, training, evaluation, pusher, monitoring
  extraction/                 # candidate draw and CCS profit builders
  pipeline/                   # training and batch scoring pipelines
  serving/                    # FastAPI app, schemas, scoring routes
  utils/                      # common.py, mlflow_utils.py, mongodb.py

configs/
  config.yaml                 # main full-cycle/training config
  candidate_extraction.yaml   # Mongo -> candidate draw parquet config
  ccs_profit.yaml             # Mongo -> daily CCS profit parquet config
  batch_scoring.yaml          # promoted-artifact batch scoring config
  schema.yaml                 # data schema

scripts/
  run_full_cycle.py           # candidate refresh -> CCS refresh -> train -> promote -> batch score
  extract_candidate_draws.py  # component-level candidate parquet refresh
  build_ccs_profit.py         # component-level CCS profit refresh
  run_training.py             # training-only wrapper
  run_batch_scoring.py        # batch scoring wrapper

orchestration/
  flows/full_cycle_flow.py    # Prefect-compatible full-cycle wrapper
  flows/training_flow.py      # Prefect-compatible training-only wrapper
  flows/batch_scoring_flow.py # Prefect-compatible scoring wrapper
  prefect.yaml                # self-hosted Prefect deployment spec
```

## Configuration

Secrets go in `.env`; tunable settings live in `configs/`.

| File | Purpose |
|---|---|
| `configs/config.yaml` | Full-cycle/training settings, candidate window, MLflow |
| `configs/candidate_extraction.yaml` | Candidate draw extraction from MongoDB |
| `configs/ccs_profit.yaml` | Daily CCS profit extraction from MongoDB |
| `configs/batch_scoring.yaml` | Batch scoring from `artifacts/current` |
| `.env` | MongoDB URI, MLflow tracking URI, collection names |

Candidate extraction resumes/skips existing partitions by default. CCS profit refresh skips existing days by default and rebuilds only when `--force-ccs` is used through the full-cycle command or `--force` is used on the component script.

## Outputs

After a successful full-cycle run:

- `artifacts/full_cycle_runs/<full_cycle_id>/candidate_extraction_summary.json`
- `artifacts/full_cycle_runs/<full_cycle_id>/ccs_profit_summary.json`
- `artifacts/full_cycle_runs/<full_cycle_id>/full_cycle_summary.json`
- `artifacts/runs/<run_id>/` training, evaluation, monitoring, and metadata outputs
- `artifacts/current/` promoted serving bundle and batch scoring outputs

Full-cycle runs log candidate extraction, CCS refresh, training, evaluation, monitoring, promotion, and batch scoring artifacts into one top-level MLflow run.

## Serving and Streamlit

```powershell
python scripts/run_api.py --config configs/config.yaml --host 127.0.0.1 --port 8000
$env:FRAUD_API_BASE_URL='http://127.0.0.1:8000'
streamlit run streamlit_partnership_demo.py
```

The API serves promoted artifacts from `artifacts/current`. Streamlit calls the API for draw, member, alert, and CCS review surfaces.

## Monitoring

After each training run, Evidently drift reports are written under `artifacts/runs/<run_id>/monitoring/` when enabled. Monitoring is advisory and does not block promotion.

## Orchestration

The flows can run as plain Python scripts without a Prefect server:

```powershell
python orchestration/flows/full_cycle_flow.py --config configs/config.yaml
python orchestration/flows/training_flow.py --config configs/config.yaml
python orchestration/flows/batch_scoring_flow.py --config configs/batch_scoring.yaml
```

For scheduled runs, use the self-hosted Prefect deployment in `orchestration/prefect.yaml`:

```powershell
pip install "prefect>=2.20.0,<3.0.0"
prefect work-pool create fraud-pool --type process
prefect deploy --prefect-file orchestration/prefect.yaml
prefect worker start --pool fraud-pool
```

`full-cycle-weekly` is the scheduled production path. `training-weekly` is retained as a disabled training-only component deployment.

## Artifact Retention

```powershell
# Dry run
python scripts/cleanup_old_runs.py --keep 5

# Execute cleanup
python scripts/cleanup_old_runs.py --keep 5 --execute
```

## Out of Scope

Polars migration, DuckDB integration, model redesign, Kubernetes, Prometheus/Grafana, CI/CD, and DVC are not part of this implementation pass.
