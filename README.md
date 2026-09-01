# Fraud Detection

Python services and reporting workflows for fraud detection.

## Local setup

1. Activate the virtual environment: `./.venv/Scripts/Activate.ps1`
2. Install the project and development tools: `python -m pip install -e . --group dev`
3. Copy `.env.example` to `.env` and set environment-specific values.
4. Run checks: `ruff check .` and `pytest`.

## Layout

- `src/fraud_detection/`: application package
- `tests/`: automated tests