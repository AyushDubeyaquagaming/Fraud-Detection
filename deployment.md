# Deployment Guide

This project runs as a Docker Compose stack with FastAPI, Streamlit, MLflow, Prefect, and a Prefect worker.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2
- Git access to this repository
- MongoDB credentials for the source roulette collection
- Enough disk space for `artifacts/`, `data_store/`, `data_cache/`, and Docker volumes

## 1. Prepare Environment

Clone the repository and enter the project folder:

```powershell
git clone <repo-url>
cd Fraud-Detection
git checkout model/feature-exploration
```

Create the runtime env file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill at least:

```text
MONGODB_URI=
MONGODB_DATABASE=gk-reports
MONGODB_COLLECTION_ROULETTE_REPORT=roulette_game_report
```

Keep the default local service URLs unless ports `8000`, `8501`, `5000`, or `4200` are already used.

## 2. Build Images

```powershell
docker compose build
```

## 3. Start Services

```powershell
docker compose up -d mlflow-server prefect-postgres prefect-server fraud-detection-api prefect-worker streamlit-ui
```

Open:

```text
FastAPI:   http://localhost:8000/docs
Streamlit: http://localhost:8501
MLflow:    http://localhost:5000
Prefect:   http://localhost:4200
```

## 4. Deploy Prefect Flows

After Prefect is healthy, register the flows:

```powershell
docker compose run --rm fraud-detection deploy-flows
```

## 5. Run Pipeline

Run the full training and scoring cycle:

```powershell
docker compose run --rm fraud-detection full-cycle
```

This can take several hours depending on MongoDB volume and candidate-store state.

## 6. Verify Health

```powershell
Invoke-WebRequest http://localhost:8000/health -UseBasicParsing
Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing
Invoke-WebRequest http://localhost:5000/health -UseBasicParsing
Invoke-WebRequest http://localhost:4200/api/health -UseBasicParsing
docker compose ps
```

## 7. Useful Operations

Run batch scoring only:

```powershell
docker compose run --rm fraud-detection score
```

Run training only:

```powershell
docker compose run --rm fraud-detection train
```

Reload FastAPI artifacts after a successful promoted run:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/reload
```

Stop the stack:

```powershell
docker compose down
```

Do not commit `.env`; it contains deployment secrets.
