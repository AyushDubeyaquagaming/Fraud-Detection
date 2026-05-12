#!/bin/sh
set -e

case "$1" in
  full-cycle)
    shift
    exec python scripts/run_full_cycle.py "$@"
    ;;
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
    POOL="${PREFECT_POOL_NAME:-fraud-pool}"
    # The Prefect worker does not create process pools automatically.
    if ! prefect work-pool inspect "$POOL" >/dev/null 2>&1; then
        echo "Creating Prefect work pool '$POOL' (type=process)..."
        prefect work-pool create "$POOL" --type process || true
    fi
    exec prefect worker start --pool "$POOL" --type process "$@"
    ;;
  deploy-flows)
    shift
    yes n | prefect deploy --prefect-file orchestration/prefect.yaml --all "$@"
    exit $?
    ;;
  *)
    exec "$@"
    ;;
esac
