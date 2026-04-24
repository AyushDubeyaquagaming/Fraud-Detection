from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.spatial.distance import mahalanobis

from fraud_detection.constants.constants import (
    ALERT_QUEUE_FILE,
    BATCH_SCORING_REPORT_FILE,
    BATCH_SCORING_CONFIG_FILE_PATH,
    HYBRID_EVALUATION_FILE,
    HYBRID_SCORED_PLAYERS_FILE,
    MODEL_BUNDLE_FILE,
    REPO_ROOT,
    WEEKLY_SCORING_MANIFEST_FILE,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import load_joblib, read_json, read_yaml, write_json

logger = get_logger(__name__)

BATCH_SCORING_PARQUET_BATCH_SIZE = 100_000

COHORT_SCOPE_NOTE = (
    "Scores are relative to the analysis cohort (~1,045 players), "
    "not the full BetBlitz platform."
)


def _normalize_component(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.5))


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_batch_window(ingestion_config: dict, batch_config: dict) -> dict[str, object]:
    window_cfg = dict(batch_config.get("window", {}) or {})
    if not window_cfg and ingestion_config.get("source") == "mongodb":
        mongo_cfg = ingestion_config.get("mongodb", {})
        if mongo_cfg.get("strategy") == "date_window":
            window_cfg = dict(mongo_cfg.get("strategy_params", {}) or {})
    window_cfg.setdefault("timestamp_field", "trans_date")
    return window_cfg


def _resolve_window_bounds(window_cfg: dict[str, object]) -> tuple[str, datetime | None, datetime | None, int | None]:
    timestamp_field = str(window_cfg.get("timestamp_field", "trans_date"))
    start_date = window_cfg.get("start_date")
    end_date = window_cfg.get("end_date")
    lookback_days = window_cfg.get("lookback_days")

    if start_date is not None and end_date is not None:
        start_dt = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
        end_dt = pd.Timestamp(end_date, tz="UTC").to_pydatetime()
        return timestamp_field, start_dt, end_dt, None

    if lookback_days is None:
        return timestamp_field, None, None, None

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    return timestamp_field, start_dt, end_dt, int(lookback_days)


def _filter_dataframe_to_window(
    raw_df: pd.DataFrame,
    window_cfg: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    timestamp_field, start_dt, end_dt, lookback_days = _resolve_window_bounds(window_cfg)
    metadata = {
        "timestamp_field": timestamp_field,
        "window_start": start_dt.isoformat() if start_dt is not None else None,
        "window_end": end_dt.isoformat() if end_dt is not None else None,
        "lookback_days": lookback_days,
    }

    if start_dt is None or end_dt is None:
        return raw_df, metadata

    if timestamp_field not in raw_df.columns:
        raise FraudDetectionException(
            ValueError(
                f"Configured batch scoring window column '{timestamp_field}' is missing from the input data."
            ),
            sys,
        )

    timestamps = pd.to_datetime(raw_df[timestamp_field], utc=True, errors="coerce")
    filtered = raw_df.loc[timestamps.notna() & (timestamps >= start_dt) & (timestamps < end_dt)].copy()
    return filtered, metadata


def _to_parquet_filter_bound(value: datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime()


def _to_arrow_filter_bound(value: datetime, arrow_type: pa.DataType) -> object:
    timestamp = pd.Timestamp(value)
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz:
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert(arrow_type.tz)
        else:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.to_pydatetime()
    if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.date()
    raise TypeError(f"Unsupported parquet timestamp field type for window filtering: {arrow_type}")


def _build_parquet_window_filter(
    schema: pa.Schema,
    timestamp_field: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
):
    if start_dt is None or end_dt is None:
        return None
    if timestamp_field not in schema.names:
        raise FraudDetectionException(
            ValueError(
                f"Configured batch scoring window column '{timestamp_field}' is missing from the parquet schema."
            ),
            sys,
        )

    arrow_type = schema.field(timestamp_field).type
    return (ds.field(timestamp_field) >= _to_arrow_filter_bound(start_dt, arrow_type)) & (
        ds.field(timestamp_field) < _to_arrow_filter_bound(end_dt, arrow_type)
    )


def _load_parquet_with_pyarrow_window(
    parquet_path: Path,
    timestamp_field: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    dataset = ds.dataset(parquet_path, format="parquet")
    filter_expr = _build_parquet_window_filter(dataset.schema, timestamp_field, start_dt, end_dt)
    return dataset.to_table(filter=filter_expr).to_pandas()


def _scan_dataset_to_parquet(
    dataset: ds.Dataset,
    output_path: Path,
    filter_expr,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    row_count = 0
    member_ids: set[str] = set()
    try:
        scanner = dataset.scanner(filter=filter_expr, batch_size=BATCH_SCORING_PARQUET_BATCH_SIZE)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            row_count += batch.num_rows

            if "member_id" in table.column_names:
                member_ids.update(str(value) for value in table["member_id"].to_pylist() if value is not None)
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        output_path.unlink(missing_ok=True)
        raise FraudDetectionException(
            ValueError("Requested batch scoring window returned 0 parquet rows."),
            sys,
        )

    return {
        "row_count": row_count,
        "member_count": len(member_ids),
    }


def _stage_parquet_window_to_path(
    parquet_path: str | Path,
    output_path: Path,
    window_cfg: dict[str, object],
) -> tuple[dict[str, int], dict[str, object]]:
    resolved_path = _resolve_repo_path(parquet_path)
    timestamp_field, start_dt, end_dt, lookback_days = _resolve_window_bounds(window_cfg)
    metadata = {
        "timestamp_field": timestamp_field,
        "window_start": start_dt.isoformat() if start_dt is not None else None,
        "window_end": end_dt.isoformat() if end_dt is not None else None,
        "lookback_days": lookback_days,
    }

    dataset = ds.dataset(resolved_path, format="parquet")
    filter_expr = _build_parquet_window_filter(dataset.schema, timestamp_field, start_dt, end_dt)
    stats = _scan_dataset_to_parquet(dataset, output_path, filter_expr)
    return stats, metadata


def _stage_mongodb_window_to_path(
    ingestion_config: dict,
    window_cfg: dict[str, object],
    output_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    from fraud_detection.utils.mongodb import (
        build_query_batches_from_strategy,
        stream_query_batches_to_parquet,
    )

    mongo_cfg = ingestion_config["mongodb"]
    strategy = mongo_cfg.get("strategy", "date_window")
    strategy_params = dict(mongo_cfg.get("strategy_params", {}))
    if window_cfg:
        strategy_params.update(window_cfg)
    _, start_dt, end_dt, lookback_days = _resolve_window_bounds(strategy_params)
    metadata = {
        "timestamp_field": str(strategy_params.get("timestamp_field", "trans_date")),
        "window_start": start_dt.isoformat() if start_dt is not None else None,
        "window_end": end_dt.isoformat() if end_dt is not None else None,
        "lookback_days": lookback_days,
    }

    logger.info(
        "BatchScoringPipeline: mongodb strategy=%s, params=%s",
        strategy,
        strategy_params,
    )
    query_filters = build_query_batches_from_strategy(strategy, strategy_params)
    stats = stream_query_batches_to_parquet(
        uri_env_var=mongo_cfg["uri_env_var"],
        db_env_var=mongo_cfg["database_env_var"],
        collection_env_var=mongo_cfg["collection_env_var"],
        output_paths=[output_path],
        query_filters=query_filters,
    )
    stats["strategy_used"] = strategy
    return stats, metadata


def _load_parquet_to_window(
    parquet_path: str | Path,
    window_cfg: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_path = _resolve_repo_path(parquet_path)
    timestamp_field, start_dt, end_dt, lookback_days = _resolve_window_bounds(window_cfg)
    metadata = {
        "timestamp_field": timestamp_field,
        "window_start": start_dt.isoformat() if start_dt is not None else None,
        "window_end": end_dt.isoformat() if end_dt is not None else None,
        "lookback_days": lookback_days,
    }

    if start_dt is None or end_dt is None:
        return pd.read_parquet(resolved_path), metadata

    try:
        raw_df = pd.read_parquet(
            resolved_path,
            filters=[
                (timestamp_field, ">=", _to_parquet_filter_bound(start_dt)),
                (timestamp_field, "<", _to_parquet_filter_bound(end_dt)),
            ],
        )
    except Exception as exc:
        logger.info(
            "BatchScoringPipeline: pandas parquet pushdown unavailable for %s (%s); retrying with pyarrow dataset scan",
            resolved_path,
            exc,
        )
        try:
            raw_df = _load_parquet_with_pyarrow_window(
                resolved_path,
                timestamp_field,
                start_dt,
                end_dt,
            )
        except Exception as fallback_exc:
            raise FraudDetectionException(
                RuntimeError(
                    "Batch scoring could not load the requested parquet window without a full read. "
                    f"Pandas error: {exc}. PyArrow error: {fallback_exc}"
                ),
                sys,
            ) from fallback_exc

    filtered_df, _ = _filter_dataframe_to_window(raw_df, window_cfg)
    return filtered_df, metadata


class BatchScoringPipeline:
    def __init__(self, config_path: Path = BATCH_SCORING_CONFIG_FILE_PATH):
        self.config_path = config_path

    def run(self) -> Path:
        logger.info("BatchScoringPipeline: starting")
        try:
            config = read_yaml(self.config_path)
            current_dir = _resolve_repo_path(config["pipeline"]["current_dir"])
            batch_cfg = config.get("batch_scoring", {})
            ing_cfg = config["data_ingestion"]
            mode = batch_cfg.get("mode", "operational")
            require_labels = batch_cfg.get("require_labels", False)
            op_filter_cfg = batch_cfg.get("operational_filter", {}) or {}
            op_filter_enabled = bool(op_filter_cfg.get("enabled", False))
            op_min_draws = int(op_filter_cfg.get("min_draws_played", 0))
            alert_queue_size = int(batch_cfg.get("alert_queue_size", 50))
            batch_window_cfg = _resolve_batch_window(ing_cfg, batch_cfg)
            raw_window_metadata: dict[str, object] = {
                "timestamp_field": str(batch_window_cfg.get("timestamp_field", "trans_date")),
                "window_start": None,
                "window_end": None,
                "lookback_days": batch_window_cfg.get("lookback_days"),
            }

            logger.info(
                "BatchScoringPipeline: mode=%s, require_labels=%s, "
                "operational_filter=%s (min_draws_played=%d), alert_queue_size=%d, current_dir=%s",
                mode, require_labels, op_filter_enabled, op_min_draws, alert_queue_size, current_dir,
            )

            # Load model bundle
            bundle_path = current_dir / MODEL_BUNDLE_FILE
            if not bundle_path.exists():
                raise FileNotFoundError(
                    f"Model bundle not found: {bundle_path}. Run training pipeline first."
                )
            bundle = load_joblib(bundle_path)
            logger.info("Loaded model bundle from %s", bundle_path)

            current_dir.mkdir(parents=True, exist_ok=True)
            tmp_raw_path = current_dir / "_tmp_scoring_raw.parquet"
            source_stats: dict[str, object]

            # Load data
            if ing_cfg["source"] == "parquet":
                source_stats, raw_window_metadata = _stage_parquet_window_to_path(
                    ing_cfg["parquet_path"],
                    tmp_raw_path,
                    batch_window_cfg,
                )
            else:
                source_stats, raw_window_metadata = _stage_mongodb_window_to_path(
                    ing_cfg,
                    batch_window_cfg,
                    tmp_raw_path,
                )

            logger.info(
                "Staged %d raw rows to %s for batch scoring",
                int(source_stats["row_count"]),
                tmp_raw_path,
            )

            # Feature engineering
            from fraud_detection.entity.config_entity import FeatureEngineeringConfig
            from fraud_detection.entity.artifact_entity import DataIngestionArtifact
            from fraud_detection.components.feature_engineering import FeatureEngineering

            val_cfg = config["data_validation"]
            fe_cfg = config["feature_engineering"]

            ingestion_artifact = DataIngestionArtifact(
                raw_data_path=tmp_raw_path,
                ingestion_report_path=current_dir / "_tmp_ingestion_report.json",
                row_count=int(source_stats["row_count"]),
                member_count=int(source_stats.get("member_count", 0)),
                source_type=ing_cfg["source"],
                strategy_used=str(source_stats.get("strategy_used")) if source_stats.get("strategy_used") else None,
                query_count=int(source_stats.get("query_count", 1)),
                date_range=source_stats.get("date_range"),
            )

            if mode == "replay_eval":
                fe_mode = "training_eval"
            else:
                fe_mode = "operational"

            fe_config = FeatureEngineeringConfig(
                exclude_cols=fe_cfg["exclude_cols"],
                log1p_cols=fe_cfg["log1p_cols"],
                apply_pre_fraud_cutoff=(fe_mode == "training_eval"),
                fraud_csv_path=REPO_ROOT / val_cfg["fraud_csv_path"],
                output_dir=current_dir / "_tmp_fe",
                mode=fe_mode,
            )
            fe_artifact = FeatureEngineering(fe_config, ingestion_artifact).initiate_feature_engineering()
            player_df = pd.read_parquet(fe_artifact.player_features_path)

            # Score with bundle
            from fraud_detection.components.model_training import (
                ANOMALY_COMPONENT_WEIGHTS,
                make_model_frame,
                make_style_frame,
            )

            feat_cfg_path = current_dir / "feature_pipeline_config.json"
            import json
            feature_columns = bundle.get("feature_columns", [])
            if feat_cfg_path.exists():
                with open(feat_cfg_path) as f:
                    feat_cfg = json.load(f)
                log1p_cols = bundle.get("log1p_columns") or feat_cfg.get("log1p_cols", fe_cfg["log1p_cols"])
                style_columns = bundle.get("style_columns") or feat_cfg.get("style_columns", [])
                style_log1p_cols = bundle.get("style_log1p_columns") or feat_cfg.get("style_log1p_cols", [])
            else:
                log1p_cols = bundle.get("log1p_columns") or fe_cfg["log1p_cols"]
                style_columns = bundle.get("style_columns", [])
                style_log1p_cols = bundle.get("style_log1p_columns", [])

            X_raw = make_model_frame(player_df, log1p_cols, feature_columns)

            iso_forest = bundle["iso_forest"]
            kmeans = bundle["kmeans"]
            mean_vec = bundle["mahal_stats"]["mean_vec"]
            cov_inv = bundle["mahal_stats"]["cov_inv"]
            scaler_unsup = bundle["scaler_unsup"]
            scaler_operational = bundle["scaler_operational"]
            lr_operational = bundle["lr_operational"]
            anomaly_w = float(bundle.get("anomaly_weight", 0.60))
            supervised_w = float(bundle.get("supervised_weight", 0.40))
            component_weights = bundle.get("anomaly_component_weights", ANOMALY_COMPONENT_WEIGHTS)

            X_unsup = scaler_unsup.transform(X_raw)
            iso_raw = -iso_forest.score_samples(X_unsup)
            mahal_raw = np.array([mahalanobis(row, mean_vec, cov_inv) for row in X_unsup])
            cluster_ids = kmeans.predict(X_unsup)
            cluster_raw = np.array([
                np.linalg.norm(X_unsup[i] - kmeans.cluster_centers_[cluster_ids[i]])
                for i in range(len(X_unsup))
            ])

            player_df["cluster_id"] = cluster_ids
            player_df["cluster_distance"] = cluster_raw
            player_df["iso_forest_score_norm"] = [
                _normalize_component(v, bundle["iso_min"], bundle["iso_max"]) for v in iso_raw
            ]
            player_df["mahalanobis_norm"] = [
                _normalize_component(v, bundle["mahal_min"], bundle["mahal_max"]) for v in mahal_raw
            ]
            player_df["cluster_distance_norm"] = [
                _normalize_component(v, bundle["cluster_min"], bundle["cluster_max"]) for v in cluster_raw
            ]
            player_df["anomaly_score"] = (
                float(component_weights["iso_forest_score_norm"]) * player_df["iso_forest_score_norm"]
                + float(component_weights["mahalanobis_norm"]) * player_df["mahalanobis_norm"]
                + float(component_weights["cluster_distance_norm"]) * player_df["cluster_distance_norm"]
            )

            if bundle.get("style_scaler") is not None and bundle.get("style_pca") is not None and style_columns:
                style_frame = make_style_frame(player_df, style_log1p_cols, style_columns)
                style_coords = bundle["style_pca"].transform(bundle["style_scaler"].transform(style_frame))
                player_df["style_pc1"] = style_coords[:, 0]
                player_df["style_pc2"] = style_coords[:, 1]

            if bundle.get("full_pca") is not None:
                full_coords = bundle["full_pca"].transform(X_unsup)
                player_df["pc1"] = full_coords[:, 0]
                player_df["pc2"] = full_coords[:, 1]

            X_operational = scaler_operational.transform(X_raw)
            player_df["supervised_score"] = lr_operational.predict_proba(X_operational)[:, 1]
            player_df["risk_score"] = anomaly_w * player_df["anomaly_score"] + supervised_w * player_df["supervised_score"]

            p80 = float(player_df["risk_score"].quantile(0.80))
            p95 = float(player_df["risk_score"].quantile(0.95))
            if p95 <= p80:
                p95 = float(np.nextafter(p80, np.inf))
            risk_upper = float(player_df["risk_score"].max())
            if risk_upper <= p95:
                risk_upper = float(np.nextafter(p95, np.inf))
            player_df["risk_tier"] = pd.cut(
                player_df["risk_score"],
                bins=[-0.001, p80, p95, risk_upper],
                labels=["LOW", "MEDIUM", "HIGH"],
            )
            player_df["risk_rank"] = player_df["risk_score"].rank(ascending=False, method="min").astype(int)
            player_df["risk_percentile"] = player_df["risk_score"].rank(pct=True)
            player_df["anomaly_percentile"] = player_df["anomaly_score"].rank(pct=True)
            player_df["supervised_percentile"] = player_df["supervised_score"].rank(pct=True)
            player_df["source"] = f"batch_{mode}"

            # Save full scored population BEFORE applying the operational filter so
            # downstream analyses can always see the un-filtered view if needed.
            scored_path = current_dir / HYBRID_SCORED_PLAYERS_FILE
            player_df.to_parquet(scored_path, index=False)

            # Operational pre-filter: applied at scoring time only, not during training.
            # Training still sees the full cohort so the unsupervised scaler reflects
            # the full population baseline.
            n_before_filter = len(player_df)
            n_filtered_out = 0
            if op_filter_enabled and "draws_played" in player_df.columns:
                eligible_mask = player_df["draws_played"] >= op_min_draws
                n_filtered_out = int((~eligible_mask).sum())
                alert_source = player_df.loc[eligible_mask]
                logger.info(
                    "operational_filter: kept %d / %d players (min_draws_played=%d)",
                    len(alert_source), n_before_filter, op_min_draws,
                )
            else:
                alert_source = player_df

            # Alert queue (drop label columns for operational mode)
            alert_cols = [
                "member_id", "risk_score", "risk_tier", "anomaly_score", "supervised_score",
                "draws_played", "total_staked", "avg_entropy", "template_reuse_ratio",
            ]
            if "primary_ccs_id" in player_df.columns:
                alert_cols.insert(1, "primary_ccs_id")
            alert_cols_available = [c for c in alert_cols if c in alert_source.columns]
            alert_queue = (
                alert_source.sort_values("risk_score", ascending=False)[alert_cols_available]
                .head(alert_queue_size)
                .reset_index(drop=True)
            )
            alert_path = current_dir / ALERT_QUEUE_FILE
            alert_queue.to_csv(alert_path, index=False)

            # Evaluation summary (only if labels available)
            tier_counts = player_df["risk_tier"].astype(str).value_counts()
            eval_summary: dict = {
                "mode": mode,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "total_players": len(player_df),
                "risk_tier_distribution": {
                    "LOW": int(tier_counts.get("LOW", 0)),
                    "MEDIUM": int(tier_counts.get("MEDIUM", 0)),
                    "HIGH": int(tier_counts.get("HIGH", 0)),
                },
                "risk_p80": p80,
                "risk_p95": p95,
                "lookback_days": raw_window_metadata.get("lookback_days"),
                "window_start": raw_window_metadata.get("window_start"),
                "window_end": raw_window_metadata.get("window_end"),
                "timestamp_field": raw_window_metadata.get("timestamp_field"),
                "operational_filter": {
                    "enabled": op_filter_enabled,
                    "min_draws_played": op_min_draws,
                    "players_before_filter": n_before_filter,
                    "players_after_filter": len(alert_source),
                    "players_filtered_out": n_filtered_out,
                },
                "alert_queue_size": alert_queue_size,
                "score_distribution": {
                    "mean": float(player_df["risk_score"].mean()),
                    "median": float(player_df["risk_score"].median()),
                    "p95": float(player_df["risk_score"].quantile(0.95)),
                },
                "cohort_scope_note": COHORT_SCOPE_NOTE,
            }

            if mode == "replay_eval" and "event_fraud_flag" in player_df.columns:
                labels = player_df["event_fraud_flag"].astype(int)
                n_total = len(player_df)
                total_fraud = int(labels.sum())
                base_rate = total_fraud / n_total if n_total > 0 else 0.0

                def stats_at_k(scores: pd.Series, k: int) -> dict:
                    k = max(1, min(k, n_total))
                    top_idx = scores.nlargest(k).index
                    captured = int(labels.loc[top_idx].sum())
                    cap_rate = captured / total_fraud if total_fraud > 0 else 0.0
                    precision = captured / k
                    lift = precision / base_rate if base_rate > 0 else 0.0
                    return {
                        "k": k,
                        "captured_fraud": captured,
                        "capture_rate": cap_rate,
                        "precision": precision,
                        "lift": lift,
                    }

                eval_summary["fraud_players"] = total_fraud
                eval_summary["base_rate"] = base_rate
                scores = player_df["risk_score"]
                eval_summary["capture_stats"] = {
                    "top_1pct": stats_at_k(scores, max(1, int(n_total * 0.01))),
                    "top_5pct": stats_at_k(scores, max(1, int(n_total * 0.05))),
                    "top_10pct": stats_at_k(scores, max(1, int(n_total * 0.10))),
                    "top_20pct": stats_at_k(scores, max(1, int(n_total * 0.20))),
                    "top_50": stats_at_k(scores, 50),
                    "top_500": stats_at_k(scores, 500),
                }
                # Legacy capture_rates (count-only) retained for backward compatibility.
                eval_summary["capture_rates"] = {
                    b: s["captured_fraud"] for b, s in eval_summary["capture_stats"].items()
                }

            serving_manifest_path = current_dir / "serving_manifest.json"
            serving_manifest = read_json(serving_manifest_path) if serving_manifest_path.exists() else {}
            eval_summary["source_run_id"] = serving_manifest.get("run_id")
            eval_summary["model_version"] = serving_manifest.get("model_version")

            eval_path = current_dir / HYBRID_EVALUATION_FILE
            write_json(eval_summary, eval_path)

            snapshot_manifest = {
                "snapshot_type": "weekly_serving",
                "snapshot_generated_at": eval_summary["scored_at"],
                "lookback_days": raw_window_metadata.get("lookback_days"),
                "window_start": raw_window_metadata.get("window_start"),
                "window_end": raw_window_metadata.get("window_end"),
                "timestamp_field": raw_window_metadata.get("timestamp_field"),
                "source_run_id": serving_manifest.get("run_id"),
                "model_version": serving_manifest.get("model_version"),
                "risk_p80": p80,
                "risk_p95": p95,
                "scored_players_file": HYBRID_SCORED_PLAYERS_FILE,
                "evaluation_file": HYBRID_EVALUATION_FILE,
                "alert_queue_file": ALERT_QUEUE_FILE,
                "total_players": len(player_df),
            }
            write_json(snapshot_manifest, current_dir / WEEKLY_SCORING_MANIFEST_FILE)

            # Scoring report
            scoring_report: dict = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "total_players": len(player_df),
                "lookback_days": raw_window_metadata.get("lookback_days"),
                "window_start": raw_window_metadata.get("window_start"),
                "window_end": raw_window_metadata.get("window_end"),
                "operational_filter": {
                    "enabled": op_filter_enabled,
                    "min_draws_played": op_min_draws,
                    "players_before_filter": n_before_filter,
                    "players_after_filter": len(alert_source),
                    "players_filtered_out": n_filtered_out,
                },
                "alert_queue_size": alert_queue_size,
                "alert_queue_rows": len(alert_queue),
                "source": ing_cfg["source"],
                "scored_path": str(scored_path),
                "alert_path": str(alert_path),
            }
            if source_stats.get("strategy_used"):
                scoring_report["strategy_used"] = source_stats["strategy_used"]
                scoring_report["query_count"] = int(source_stats.get("query_count", 1))
            report_path = current_dir / BATCH_SCORING_REPORT_FILE
            write_json(scoring_report, report_path)

            # Cleanup tmp files
            for tmp_f in [tmp_raw_path]:
                try:
                    tmp_f.unlink(missing_ok=True)
                except Exception:
                    pass

            logger.info(
                "BatchScoringPipeline: complete — %d players scored, alert_queue at %s",
                len(player_df), alert_path,
            )
            return current_dir

        except FraudDetectionException:
            raise
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
