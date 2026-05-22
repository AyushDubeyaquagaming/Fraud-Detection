"""Full-cycle orchestration for candidate-store training.

This flow keeps upstream data refreshes explicit while producing one MLflow
lineage run for extraction, CCS profit refresh, training, promotion, and batch
scoring.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from prefect import flow, get_run_logger

    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False

from fraud_detection.constants.constants import (
    BATCH_SCORING_CONFIG_FILE_PATH,
    CONFIG_FILE_PATH,
    MODEL_BUNDLE_FILE,
    RUN_METADATA_FILE,
)
from fraud_detection.extraction.candidate_extractor import (
    CandidateDrawExtractor,
    load_candidate_extraction_config,
    parse_utc_date,
)
from fraud_detection.extraction.ccs_profit_aggregator import build_ccs_daily_profit
from fraud_detection.pipeline.batch_scoring_pipeline import BatchScoringPipeline
from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from fraud_detection.utils.common import read_json, read_yaml, write_json
from fraud_detection.utils.mlflow_utils import (
    get_tracking_uri,
    log_artifact_safe,
    log_metrics_safe,
    log_params_safe,
    setup_mlflow,
)
from orchestration.notifications import notify_failure


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), default=str))


def _candidate_summary_dict(summary) -> dict[str, Any]:
    return _json_safe_dataclass(summary)


def _resolve_window(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    *,
    window_mode: str = "fixed",
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    if bool(start_date) != bool(end_date):
        raise ValueError("Full cycle requires both start_date and end_date when either is provided.")
    if start_date and end_date:
        return parse_utc_date(str(start_date)), parse_utc_date(str(end_date))

    window = (config.get("partnership", {}) or {}).get("candidate_window", {}) or {}
    if str(window_mode or "fixed").strip().lower() == "rolling":
        lookback_days = int(window.get("rolling_lookback_days", 90))
        if lookback_days <= 0:
            raise ValueError("partnership.candidate_window.rolling_lookback_days must be positive.")
        anchor = _coerce_utc_datetime(now or datetime.now(timezone.utc))
        end_dt = datetime.combine(anchor.date(), datetime_time.min, tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)
        return start_dt, end_dt

    start_value = window.get("start_date")
    end_value = window.get("end_date")
    if not start_value or not end_value:
        raise ValueError("Full cycle requires start_date and end_date, either as args or partnership.candidate_window.")
    return parse_utc_date(str(start_value)), parse_utc_date(str(end_value))


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ccs_end_date_for_candidate_window(end_dt: datetime):
    return (end_dt.astimezone(timezone.utc) - timedelta(microseconds=1)).date()


def _ccs_start_date_for_candidate_window(start_dt: datetime, config: dict[str, Any]) -> date:
    ccs_cfg = (config.get("partnership", {}) or {}).get("ccs_features", {}) or {}
    raw_windows = ccs_cfg.get("windows_days", [1, 7])
    if isinstance(raw_windows, (str, int, float)):
        raw_windows = [raw_windows]
    windows = []
    for window in raw_windows or []:
        try:
            parsed = int(window)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            windows.append(parsed)
    if windows:
        max_window = max(windows)
    else:
        max_window = 7
    return start_dt.date() - timedelta(days=max_window - 1)


def _start_mlflow_run(config: dict[str, Any], run_name: str):
    import mlflow

    tracking_uri = get_tracking_uri()
    experiment_name = (config.get("mlflow", {}) or {}).get("experiment_name", "fraud_detection_partnership_v1")
    exp_id = setup_mlflow(tracking_uri, experiment_name)
    mlflow.start_run(run_name=run_name, experiment_id=exp_id)
    return mlflow


def _mlflow_active(mlflow_module: Any | None) -> bool:
    if mlflow_module is None:
        return False
    try:
        return mlflow_module.active_run() is not None
    except Exception:
        return False


def _log_params_if_active(mlflow_module: Any | None, params: dict[str, Any]) -> None:
    if _mlflow_active(mlflow_module):
        log_params_safe(params)


def _log_metrics_if_active(mlflow_module: Any | None, metrics: dict[str, Any]) -> None:
    if _mlflow_active(mlflow_module):
        log_metrics_safe(metrics)


def _log_artifact_if_active(mlflow_module: Any | None, path: str | Path) -> None:
    if _mlflow_active(mlflow_module):
        log_artifact_safe(str(path))


def _write_resolved_training_config(
    config: dict[str, Any],
    *,
    full_cycle_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> Path:
    resolved = copy.deepcopy(config)
    partnership = resolved.setdefault("partnership", {})
    candidate_window = partnership.setdefault("candidate_window", {})
    candidate_window["start_date"] = start_dt.date().isoformat()
    candidate_window["end_date"] = end_dt.date().isoformat()
    config_path = full_cycle_dir / "resolved_training_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return config_path


def _batch_model_bundle_exists(batch_config_path: Path) -> bool:
    try:
        batch_config = read_yaml(batch_config_path)
    except Exception:
        return False
    current_dir = _resolve_repo_path((batch_config.get("pipeline", {}) or {}).get("current_dir", "artifacts/current"))
    return (current_dir / MODEL_BUNDLE_FILE).exists()


def run_full_cycle(
    *,
    config_path: str | Path = CONFIG_FILE_PATH,
    candidate_config_path: str | Path = "configs/candidate_extraction.yaml",
    ccs_config_path: str | Path = "configs/ccs_profit.yaml",
    batch_config_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    window_mode: str = "fixed",
    force_candidates: bool = False,
    force_ccs: bool = False,
) -> dict[str, Any]:
    config_path = _resolve_repo_path(config_path)
    candidate_config_path = _resolve_repo_path(candidate_config_path)
    ccs_config_path = _resolve_repo_path(ccs_config_path)
    config = read_yaml(config_path)
    start_dt, end_dt = _resolve_window(config, start_date, end_date, window_mode=window_mode)
    full_cycle_id = f"full_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full_cycle_dir = _resolve_repo_path(config["pipeline"]["artifact_root"]) / "full_cycle_runs" / full_cycle_id
    full_cycle_dir.mkdir(parents=True, exist_ok=True)

    mlflow = None
    mlflow_started = False
    result: dict[str, Any] = {
        "status": "STARTED",
        "full_cycle_id": full_cycle_id,
        "config_path": str(config_path),
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "window_mode": window_mode,
        "full_cycle_dir": str(full_cycle_dir),
    }
    try:
        try:
            mlflow = _start_mlflow_run(config, full_cycle_id)
            mlflow_started = True
            mlflow.set_tag("run_id", full_cycle_id)
            mlflow.set_tag("source", "full_cycle")
            mlflow.set_tag(
                "candidate_store_mode",
                str(bool((config.get("partnership", {}) or {}).get("use_candidate_store", False))).lower(),
            )
            mlflow.set_tag("candidate_window_start", start_dt.date().isoformat())
            mlflow.set_tag("candidate_window_end", end_dt.date().isoformat())
            _log_params_if_active(
                mlflow,
                {
                    "config_path": str(config_path),
                    "candidate_config_path": str(candidate_config_path),
                    "ccs_config_path": str(ccs_config_path),
                    "start_date": start_dt.isoformat(),
                    "end_date": end_dt.isoformat(),
                    "window_mode": window_mode,
                    "force_candidates": force_candidates,
                    "force_ccs": force_ccs,
                },
            )
        except Exception as exc:
            result["mlflow_warning"] = str(exc)

        candidate_config = load_candidate_extraction_config(candidate_config_path)
        candidate_summary = CandidateDrawExtractor(candidate_config).run(
            start_date=start_dt,
            end_date=end_dt,
            force=force_candidates,
        )
        candidate_report_path = full_cycle_dir / "candidate_extraction_summary.json"
        write_json(_candidate_summary_dict(candidate_summary), candidate_report_path)
        result["candidate_extraction"] = _candidate_summary_dict(candidate_summary)
        _log_artifact_if_active(mlflow, candidate_report_path)
        _log_metrics_if_active(
            mlflow,
            {
                "candidate_chunks_total": candidate_summary.chunks_total,
                "candidate_chunks_succeeded": candidate_summary.succeeded,
                "candidate_chunks_skipped": candidate_summary.skipped,
                "candidate_rows_written": candidate_summary.total_rows,
                "candidate_elapsed_seconds": candidate_summary.elapsed_seconds,
            },
        )
        if candidate_summary.exit_code:
            raise RuntimeError(f"Candidate extraction failed for {candidate_summary.failed} chunk(s).")

        ccs_config = read_yaml(ccs_config_path)
        ccs_output = ccs_config.get("output", {}) or {}
        ccs_mongo = ccs_config.get("mongo", {}) or {}
        ccs_extraction = ccs_config.get("extraction", {}) or {}
        ccs_report_path = full_cycle_dir / "ccs_profit_summary.json"
        ccs_start_date = _ccs_start_date_for_candidate_window(start_dt, config)
        ccs_end_date = _ccs_end_date_for_candidate_window(end_dt)
        result["ccs_profit_start_date"] = ccs_start_date.isoformat()
        result["ccs_profit_end_date"] = ccs_end_date.isoformat()
        ccs_summary = build_ccs_daily_profit(
            ccs_start_date,
            ccs_end_date,
            str(ccs_output.get("base_path", "data_store/ccs_daily_profit")),
            timestamp_field=str(ccs_extraction.get("timestamp_field", "trans_date")),
            compression=str(ccs_output.get("parquet_compression", "zstd")),
            uri_env_var=str(ccs_mongo.get("uri_env_var", "MONGODB_URI")),
            database_env_var=str(ccs_mongo.get("database_env_var", "MONGODB_DATABASE")),
            collection_env_var=str(ccs_mongo.get("collection_env_var", "MONGODB_COLLECTION_ROULETTE_REPORT")),
            force=force_ccs,
            report_path=ccs_report_path,
        )
        result["ccs_profit"] = ccs_summary.to_dict()
        _log_artifact_if_active(mlflow, ccs_report_path)
        _log_metrics_if_active(
            mlflow,
            {
                "ccs_days_processed": ccs_summary.days_processed,
                "ccs_days_succeeded": ccs_summary.days_succeeded,
                "ccs_days_skipped": ccs_summary.skipped_days,
                "ccs_rows_written": ccs_summary.rows_written,
                "ccs_elapsed_seconds": ccs_summary.elapsed_seconds,
            },
        )

        training_config_path = _write_resolved_training_config(
            config,
            full_cycle_dir=full_cycle_dir,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        result["training_config_path"] = str(training_config_path)
        _log_artifact_if_active(mlflow, training_config_path)
        run_dir = TrainingPipeline(
            config_path=training_config_path,
            manage_mlflow=False,
            run_batch_scoring_on_promotion=False,
            mlflow_source="full_cycle",
        ).run()
        run_metadata_path = run_dir / RUN_METADATA_FILE
        run_metadata = read_json(run_metadata_path) if run_metadata_path.exists() else {"run_dir": str(run_dir)}
        result["training"] = run_metadata
        _log_artifact_if_active(mlflow, run_metadata_path)

        promoted = bool(run_metadata.get("promoted", False))
        if _mlflow_active(mlflow):
            mlflow.set_tag("promoted", str(promoted).lower())

        batch_path = _resolve_repo_path(
            batch_config_path
            or config.get("pipeline", {}).get("weekly_serving_snapshot_config", BATCH_SCORING_CONFIG_FILE_PATH)
        )
        if _batch_model_bundle_exists(batch_path):
            output_dir = BatchScoringPipeline(config_path=batch_path).run()
            report_path = output_dir / "batch_scoring_report.json"
            batch_result = read_json(report_path) if report_path.exists() else {"output_dir": str(output_dir)}
            _log_artifact_if_active(mlflow, report_path)
            _log_metrics_if_active(mlflow, {"batch_draws_scored": int(batch_result.get("draws_scored", 0) or 0)})
        else:
            batch_result = {
                "status": "SKIPPED_NO_CURRENT_MODEL",
                "reason": f"No model bundle found for batch config {batch_path}",
                "config_path": str(batch_path),
            }
        result["batch_scoring"] = batch_result
        result["status"] = "FINISHED"
        write_json(result, full_cycle_dir / "full_cycle_summary.json")
        _log_artifact_if_active(mlflow, full_cycle_dir / "full_cycle_summary.json")
        if mlflow_started and _mlflow_active(mlflow):
            mlflow.end_run(status="FINISHED")
        return result
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        write_json(result, full_cycle_dir / "full_cycle_summary.json")
        if mlflow_started and _mlflow_active(mlflow):
            try:
                _log_artifact_if_active(mlflow, full_cycle_dir / "full_cycle_summary.json")
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        raise


if _PREFECT_AVAILABLE:

    @flow(name="fraud-detection-full-cycle", log_prints=True)
    def full_cycle_flow(
        config_path: str = str(CONFIG_FILE_PATH),
        candidate_config_path: str = "configs/candidate_extraction.yaml",
        ccs_config_path: str = "configs/ccs_profit.yaml",
        batch_config_path: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        window_mode: str = "fixed",
        force_candidates: bool = False,
        force_ccs: bool = False,
    ) -> dict[str, Any]:
        logger = get_run_logger()
        try:
            output = run_full_cycle(
                config_path=config_path,
                candidate_config_path=candidate_config_path,
                ccs_config_path=ccs_config_path,
                batch_config_path=batch_config_path,
                start_date=start_date,
                end_date=end_date,
                window_mode=window_mode,
                force_candidates=force_candidates,
                force_ccs=force_ccs,
            )
            logger.info("Full cycle complete: %s", json.dumps(output, default=str))
            return output
        except Exception as exc:
            notify_failure(flow_name="fraud-detection-full-cycle", error=str(exc))
            raise

else:

    def full_cycle_flow(**kwargs) -> dict[str, Any]:  # type: ignore[misc]
        return run_full_cycle(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full candidate-store training cycle")
    parser.add_argument("--config", default=str(CONFIG_FILE_PATH))
    parser.add_argument("--candidate-config", default="configs/candidate_extraction.yaml")
    parser.add_argument("--ccs-config", default="configs/ccs_profit.yaml")
    parser.add_argument("--batch-config", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--window-mode", choices=["fixed", "rolling"], default="fixed")
    parser.add_argument("--force-candidates", action="store_true")
    parser.add_argument("--force-ccs", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            full_cycle_flow(
                config_path=args.config,
                candidate_config_path=args.candidate_config,
                ccs_config_path=args.ccs_config,
                batch_config_path=args.batch_config,
                start_date=args.start_date,
                end_date=args.end_date,
                window_mode=args.window_mode,
                force_candidates=args.force_candidates,
                force_ccs=args.force_ccs,
            ),
            indent=2,
            default=str,
        )
    )
