#!/bin/sh
set -e

case "$1" in
  train)
    shift
    exec python scripts/run_training.py "$@"
    ;;
  score)
    shift
    exec python scripts/run_batch_scoring.py "$@"
    ;;
  audit)
    shift
    exec python scripts/audit_artifacts.py "$@"
    ;;
  test)
    shift
    exec python -m pytest tests/ "$@"
    ;;
  shell)
    exec /bin/sh
    ;;
  serve)
    shift
    exec python scripts/run_api.py --host 0.0.0.0 --port "${API_PORT:-8000}" "$@"
    ;;
  worker)
    shift
    exec prefect worker start --pool "${PREFECT_POOL_NAME:-fraud-pool}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
