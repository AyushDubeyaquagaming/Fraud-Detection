from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fraud_detection.components.ccs_features import attach_ccs_concentration_features
from fraud_detection.components.analyst_label_overlay import apply_pair_label_overrides
from fraud_detection.components.clique_scan import CliqueRuleConfig
from fraud_detection.components.pair_scan import PairRuleConfig
from fraud_detection.components.partnership_features import (
    STAGE1_FEATURE_COLUMNS,
    PartnershipThresholds,
    add_rolling_context,
    compute_partnership_features,
    compute_partnership_features_from_candidates,
    ensure_stage1_schema,
)
from fraud_detection.entity.artifact_entity import DataIngestionArtifact, FeatureEngineeringArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, save_parquet, write_json
from fraud_detection.utils.time_utils import TIMESTAMP_CANDIDATES, coerce_mongo_datetime, normalize_event_timestamp

logger = get_logger(__name__)

FEATURE_ENGINEERING_RAW_COLUMNS = [
    "member_id",
    "draw_id",
    "bets",
    "win_points",
    "total_bet_amount",
    "session_id",
    "ccs_id",
    "createdAt",
    "updatedAt",
    "trans_date",
]
PARTNERSHIP_STREAM_BATCH_SIZE = 100_000
PARTNERSHIP_REQUIRED_COLUMNS = [
    "member_id",
    "draw_id",
    "bets",
    "win_points",
    "total_bet_amount",
    "trans_date",
]


class _ParquetFrameWriter:
    def __init__(self, path: Path):
        self.path = path
        self.schema = None
        self.writer = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.schema is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(str(self.path), self.schema)
        else:
            table = table.cast(self.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def parse_bets(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def normalize_bet_position(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"", "NAN", "NONE"}:
        return None
    if raw in {"00", "000", "DOUBLE_ZERO", "DOUBLEZERO"}:
        return "00"
    if raw.isdigit():
        number = int(raw)
        if number == 0:
            return "0"
        if 1 <= number <= 36:
            return str(number)
    return raw


def _coerce_datetime(value: Any):
    return coerce_mongo_datetime(value)


def _normalize_timestamp(df: pd.DataFrame) -> pd.Series:
    return normalize_event_timestamp(df, TIMESTAMP_CANDIDATES)


class FeatureEngineering:
    """Build partnership-first feature artifacts for Phase B."""

    def __init__(self, config: FeatureEngineeringConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact
        analyst_cfg = self.config.partnership.get("analyst_label_overlay", {}) or {}
        self._analyst_label_overlay_enabled = bool(analyst_cfg.get("enabled", False))
        self._analyst_any_member_clique_override = bool(analyst_cfg.get("any_member_clique_override", False))
        self._analyst_label_overlay_unavailable = False
        self._synthetic_positives_from_strict_pattern = bool(
            self.config.partnership.get("synthetic_positives_from_strict_pattern", False)
        )

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        logger.info("PartnershipFeatureEngineering: starting")
        try:
            ensure_dir(self.config.output_dir)
            if self.config.partnership.get("use_candidate_store", False):
                return self._initiate_candidate_store_feature_engineering()
            thresholds = PartnershipThresholds(**dict(self.config.partnership.get("candidate_thresholds", {})))
            stage1_base, pair_events, partnership_table = self._build_stage1_base_from_parquet(
                self.ingestion_artifact.raw_data_path,
                thresholds,
            )
            stage1_features = add_rolling_context(stage1_base, partnership_table) if not stage1_base.empty else stage1_base
            stage1_features = ensure_stage1_schema(stage1_features)
            stage1_labeled = self._attach_labels(stage1_features)

            paths = {
                "stage1_features": self.config.output_dir / "stage1_features.parquet",
                "stage1_labels": self.config.output_dir / "stage1_labels.parquet",
                "partnership_table": self.config.output_dir / "partnership_table.parquet",
                "pair_events": self.config.output_dir / "pair_events.parquet",
                "summary": self.config.output_dir / "feature_summary.json",
            }
            save_parquet(stage1_features, paths["stage1_features"])
            save_parquet(stage1_labeled, paths["stage1_labels"])
            save_parquet(partnership_table, paths["partnership_table"])
            save_parquet(pair_events, paths["pair_events"])
            summary = {
                "mode": self.config.mode,
                "raw_rows": int(self.ingestion_artifact.row_count),
                "stage1_rows": int(len(stage1_features)),
                "pair_events": int(len(pair_events)),
                "fraud_rows": int(stage1_labeled["label_gold"].sum()),
                "feature_columns": STAGE1_FEATURE_COLUMNS,
                "approach": "partnership_v1",
            }
            write_json(summary, paths["summary"])
            return FeatureEngineeringArtifact(
                player_features_path=paths["stage1_features"],
                history_df_path=self.ingestion_artifact.raw_data_path,
                fraud_player_count=int(stage1_labeled.loc[stage1_labeled["label_gold"].eq(1), "member_id"].nunique()),
                dropped_positive_count=0,
                feature_columns=STAGE1_FEATURE_COLUMNS,
                feature_summary_path=paths["summary"],
                mode=self.config.mode,
                stage1_features_path=paths["stage1_features"],
                stage1_labels_path=paths["stage1_labels"],
                partnership_table_path=paths["partnership_table"],
                pair_events_path=paths["pair_events"],
            )
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc

    def _initiate_candidate_store_feature_engineering(self) -> FeatureEngineeringArtifact:
        pair_rule_values = dict(self.config.partnership.get("pair_rules", {}))
        if "strict_inference_filter" in self.config.partnership:
            pair_rule_values["strict_inference_filter"] = self.config.partnership["strict_inference_filter"]
        pair_rules = PairRuleConfig(**pair_rule_values)
        clique_rules = CliqueRuleConfig(**dict(self.config.partnership.get("clique_rules", {})))
        paths = {
            "stage1_features": self.config.output_dir / "stage1_features.parquet",
            "stage1_labels": self.config.output_dir / "stage1_labels.parquet",
            "partnership_table": self.config.output_dir / "partnership_table.parquet",
            "pair_events": self.config.output_dir / "pair_events.parquet",
            "ccs_concentration_table": self.config.output_dir / "ccs_concentration_table.parquet",
            "summary": self.config.output_dir / "feature_summary.json",
        }
        stage1_writer = _ParquetFrameWriter(paths["stage1_features"])
        stage1_labels_writer = _ParquetFrameWriter(paths["stage1_labels"])
        pair_events_writer = _ParquetFrameWriter(paths["pair_events"])
        candidate_draw_rows = 0
        pair_rows = 0
        strict_pairs = 0
        nearmiss_pairs = 0
        sampled_negative_pairs = 0
        stage1_rows = 0
        fraud_pairs = 0
        analyst_positive_overrides = 0
        analyst_negative_overrides = 0
        try:
            for batch_number, candidate_chunk in enumerate(self._iter_candidate_store_batches(), start=1):
                batch_started = time.perf_counter()
                candidate_players = int(
                    pd.to_numeric(
                        candidate_chunk.get("qualifying_player_count", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )
                logger.info(
                    "FeatureEngineering candidate batch %d: candidate_draw_rows=%d qualifying_players=%d",
                    batch_number,
                    len(candidate_chunk),
                    candidate_players,
                )
                candidate_draw_rows += len(candidate_chunk)
                stage1_chunk, pair_chunk, _ = compute_partnership_features_from_candidates(
                    candidate_chunk,
                    rule_config=pair_rules,
                    clique_rule_config=clique_rules,
                    mode="training",
                    ordinary_negative_sample=int(self.config.partnership.get("emit_negatives_sample", 5)),
                    rolling_context=False,
                    random_seed=int(self.config.partnership.get("random_seed", 42)),
                )
                stage1_chunk = self._attach_ccs_features(stage1_chunk)
                analyst_labels = self._read_analyst_labels(
                    draw_ids=pair_chunk.get("draw_id", pd.Series(dtype=object)).dropna().tolist()
                )
                labeled_chunk = self._attach_pair_labels(pair_chunk, analyst_labels=analyst_labels)
                stage1_writer.write(stage1_chunk)
                stage1_labels_writer.write(labeled_chunk)
                pair_events_writer.write(pair_chunk)
                pair_rows += int(len(pair_chunk))
                strict_pairs += int(pd.to_numeric(labeled_chunk.get("is_strict_match"), errors="coerce").fillna(0).sum())
                nearmiss_pairs += int(pd.to_numeric(labeled_chunk.get("is_nearmiss"), errors="coerce").fillna(0).sum())
                sampled_negative_pairs += int(pd.to_numeric(labeled_chunk.get("sampled_negative"), errors="coerce").fillna(0).sum())
                stage1_rows += int(len(stage1_chunk))
                fraud_pairs += int(pd.to_numeric(labeled_chunk.get("label_gold"), errors="coerce").fillna(0).sum())
                analyst_positive_overrides += int(labeled_chunk.get("label_source", pd.Series(dtype=object)).eq("derived_pair_analyst").sum())
                analyst_negative_overrides += int(labeled_chunk.get("label_source", pd.Series(dtype=object)).eq("analyst_not_fraud_pair").sum())
                logger.info(
                    "FeatureEngineering candidate batch %d complete: pair_rows=%d stage1_rows=%d elapsed=%.2fs",
                    batch_number,
                    len(pair_chunk),
                    len(stage1_chunk),
                    time.perf_counter() - batch_started,
                )
                del candidate_chunk, stage1_chunk, pair_chunk, labeled_chunk
                gc.collect()
                pa.default_memory_pool().release_unused()
        finally:
            stage1_writer.close()
            stage1_labels_writer.close()
            pair_events_writer.close()

        if not paths["stage1_features"].exists():
            save_parquet(ensure_stage1_schema(pd.DataFrame()), paths["stage1_features"])
        if not paths["stage1_labels"].exists():
            save_parquet(pd.DataFrame(), paths["stage1_labels"])
        if not paths["pair_events"].exists():
            save_parquet(pd.DataFrame(), paths["pair_events"])
        if not paths["ccs_concentration_table"].exists():
            save_parquet(pd.DataFrame(), paths["ccs_concentration_table"])
        save_parquet(pd.DataFrame(), paths["partnership_table"])

        summary = {
            "mode": self.config.mode,
            "input_mode": "candidate_store",
            "candidate_draw_rows": int(candidate_draw_rows),
            "pair_rows": int(pair_rows),
            "strict_pairs": int(strict_pairs),
            "nearmiss_pairs": int(nearmiss_pairs),
            "sampled_negative_pairs": int(sampled_negative_pairs),
            "stage1_rows": int(stage1_rows),
            "fraud_pairs": int(fraud_pairs),
            "analyst_positive_pair_overrides": int(analyst_positive_overrides),
            "analyst_negative_pair_overrides": int(analyst_negative_overrides),
            "feature_columns": STAGE1_FEATURE_COLUMNS,
            "approach": "partnership_v1_candidate_store",
        }
        write_json(summary, paths["summary"])
        return FeatureEngineeringArtifact(
            player_features_path=paths["stage1_features"],
            history_df_path=self.ingestion_artifact.raw_data_path,
            fraud_player_count=0,
            dropped_positive_count=0,
            feature_columns=STAGE1_FEATURE_COLUMNS,
            feature_summary_path=paths["summary"],
            mode=self.config.mode,
            stage1_features_path=paths["stage1_features"],
            stage1_labels_path=paths["stage1_labels"],
            partnership_table_path=paths["partnership_table"],
            pair_events_path=paths["pair_events"],
            ccs_concentration_table_path=paths["ccs_concentration_table"],
        )

    def _attach_ccs_features(self, stage1_chunk: pd.DataFrame) -> pd.DataFrame:
        ccs_cfg = self.config.partnership.get("ccs_features", {}) or {}
        if not ccs_cfg.get("enabled", False):
            return stage1_chunk
        return attach_ccs_concentration_features(
            stage1_chunk,
            ccs_profit_path=ccs_cfg.get("profit_path", "data_store/ccs_daily_profit"),
            windows_days=list(ccs_cfg.get("windows_days", [1, 7])),
            concentration_threshold=float(ccs_cfg.get("concentration_threshold", 0.70)),
        )

    def _iter_candidate_store_batches(self):
        store_path = Path(self.config.partnership.get("candidate_store_path", self.ingestion_artifact.raw_data_path))
        if not store_path.is_absolute():
            from fraud_detection.constants.constants import REPO_ROOT

            store_path = REPO_ROOT / store_path
        if not store_path.exists():
            window = self.config.partnership.get("candidate_window", {})
            raise ValueError(
                f"candidate store missing for window {window.get('start_date')}..{window.get('end_date')}; "
                "run scripts/extract_candidate_draws.py first"
            )
        dataset = ds.dataset(store_path, format="parquet", partitioning="hive")
        window = self.config.partnership.get("candidate_window", {}) or {}
        batch_size = int(self.config.partnership.get("stream_batch_size", PARTNERSHIP_STREAM_BATCH_SIZE))
        filters = []
        if window.get("start_date"):
            filters.append(ds.field("trans_date_min") >= pd.Timestamp(window["start_date"], tz="UTC").to_pydatetime())
        if window.get("end_date"):
            filters.append(ds.field("trans_date_min") < pd.Timestamp(window["end_date"], tz="UTC").to_pydatetime())
        filter_expr = None
        for expr in filters:
            filter_expr = expr if filter_expr is None else filter_expr & expr
        scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)
        saw_rows = False
        for batch in scanner.to_batches():
            chunk = batch.to_pandas()
            if chunk.empty:
                continue
            saw_rows = True
            yield chunk
        if not saw_rows:
            raise ValueError(
                f"candidate store missing for window {window.get('start_date')}..{window.get('end_date')}; "
                "run scripts/extract_candidate_draws.py first"
            )

    def _attach_pair_labels(
        self,
        pair_df: pd.DataFrame,
        *,
        analyst_labels: list[dict[str, Any]] | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        labeled = pair_df.copy()
        labeled["label_stage1"] = pd.to_numeric(labeled.get("is_strict_match", 0), errors="coerce").fillna(0).astype(int)
        labeled["label_gold"] = 0
        labeled["label_source"] = "strict_rule"
        labeled["sample_weight"] = 1.0
        labeled.loc[labeled["is_nearmiss"].eq(1), "label_source"] = "nearmiss_negative"
        labeled.loc[labeled["sampled_negative"].eq(1), "label_source"] = "sampled_negative"
        if analyst_labels is None:
            analyst_labels = self._read_analyst_labels(
                draw_ids=labeled.get("draw_id", pd.Series(dtype=object)).dropna().tolist()
            )
        if self._synthetic_positives_from_strict_pattern:
            strict_signal = pd.to_numeric(
                labeled.get("is_strict_collusion_pattern", labeled.get("is_strict_match", 0)),
                errors="coerce",
            ).fillna(0).astype(int)
            pair_net_per_stake = pd.to_numeric(labeled.get("pair_net_per_stake"), errors="coerce").fillna(-1.0)
            synthetic_mask = strict_signal.eq(1) & pair_net_per_stake.ge(
                float(self.config.partnership.get("synthetic_positive_min_pair_net_per_stake", 0.0))
            )
            labeled.loc[synthetic_mask, "label_stage1"] = 1
            labeled.loc[synthetic_mask, "label_gold"] = 1
            labeled.loc[synthetic_mask, "label_source"] = "synthetic_strict_pattern"
            labeled.loc[synthetic_mask, "sample_weight"] = 1.0
        labeled, _ = apply_pair_label_overrides(
            labeled,
            analyst_labels,
            any_member_clique_override=self._analyst_any_member_clique_override,
        )
        return labeled

    def _read_analyst_labels(self, draw_ids: list[int] | None = None) -> list[dict[str, Any]]:
        if not self._analyst_label_overlay_enabled or self._analyst_label_overlay_unavailable:
            return []
        try:
            from fraud_detection.utils.mongo_predictions import read_analyst_labels_for_draws

            return read_analyst_labels_for_draws(draw_ids=draw_ids)
        except Exception as exc:
            self._analyst_label_overlay_unavailable = True
            logger.warning(
                "Analyst labels unavailable during candidate pair labeling; proceeding without analyst label overlay for the rest of this feature engineering run: %s",
                exc,
            )
            return []

    def _build_stage1_base_from_parquet(
        self,
        parquet_path: Path,
        thresholds: PartnershipThresholds,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        timestamp_field = str(self.config.partnership.get("timestamp_field", "trans_date"))
        batch_size = int(self.config.partnership.get("stream_batch_size", PARTNERSHIP_STREAM_BATCH_SIZE))
        dataset = ds.dataset(parquet_path, format="parquet")
        projected_columns = [column for column in PARTNERSHIP_REQUIRED_COLUMNS if column in dataset.schema.names]
        if timestamp_field in dataset.schema.names and timestamp_field not in projected_columns:
            projected_columns.append(timestamp_field)
        if "draw_id" not in projected_columns:
            raise ValueError("draw_id column is required for partnership feature engineering.")

        stage1_base_path = self.config.output_dir / "_stage1_base.parquet"
        pair_events_path = self.config.output_dir / "_pair_events_base.parquet"
        stage1_writer = _ParquetFrameWriter(stage1_base_path)
        pair_events_writer = _ParquetFrameWriter(pair_events_path)
        pending = pd.DataFrame(columns=projected_columns)

        def flush_chunk(chunk: pd.DataFrame) -> None:
            if chunk.empty:
                return
            stage1_chunk, pair_events_chunk, _ = compute_partnership_features(
                chunk,
                candidate_thresholds=thresholds,
                rolling_context=False,
                candidate_only=True,
                timestamp_field=timestamp_field,
            )
            stage1_writer.write(stage1_chunk)
            pair_events_writer.write(pair_events_chunk)

        try:
            scanner = dataset.scanner(columns=projected_columns, batch_size=batch_size)
            for batch in scanner.to_batches():
                chunk = batch.to_pandas()
                if chunk.empty:
                    continue
                if not pending.empty:
                    chunk = pd.concat([pending, chunk], ignore_index=True)

                draw_ids = pd.to_numeric(chunk["draw_id"], errors="coerce")
                chunk = chunk.loc[draw_ids.notna()].copy()
                if chunk.empty:
                    pending = pd.DataFrame(columns=projected_columns)
                    continue
                chunk["draw_id"] = draw_ids.loc[chunk.index].astype(int)
                last_draw_id = int(chunk["draw_id"].iloc[-1])
                flush_chunk(chunk.loc[chunk["draw_id"].ne(last_draw_id)].copy())
                pending = chunk.loc[chunk["draw_id"].eq(last_draw_id)].copy()
            flush_chunk(pending)
        finally:
            stage1_writer.close()
            pair_events_writer.close()

        stage1_base = pd.read_parquet(stage1_base_path) if stage1_base_path.exists() else ensure_stage1_schema(pd.DataFrame())
        pair_events = pd.read_parquet(pair_events_path) if pair_events_path.exists() else pd.DataFrame()
        return stage1_base, pair_events, pd.DataFrame()

    def _attach_labels(self, stage1_features: pd.DataFrame) -> pd.DataFrame:
        labeled = stage1_features.copy()
        labeled["label_gold"] = 0
        labeled["label_silver"] = 0
        labeled["label_bronze"] = 0
        labeled["label_source"] = "heuristic"
        labeled["label_tier"] = "bronze"
        if Path(self.config.fraud_csv_path).exists():
            fraud_df = pd.read_csv(self.config.fraud_csv_path)
            fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]
            if {"member_id", "draw_id"}.issubset(fraud_df.columns):
                keys = set(
                    zip(
                        fraud_df["member_id"].astype(str).str.strip().str.upper(),
                        pd.to_numeric(fraud_df["draw_id"], errors="coerce").astype("Int64"),
                    )
                )
                draw_ids = pd.to_numeric(labeled["draw_id"], errors="coerce").astype("Int64")
                labeled["label_gold"] = [
                    int((member, draw_id) in keys)
                    for member, draw_id in zip(labeled["member_id"].astype(str).str.upper(), draw_ids)
                ]
                labeled.loc[labeled["label_gold"].eq(1), "label_source"] = "csv"
                labeled.loc[labeled["label_gold"].eq(1), "label_tier"] = "gold"
        labeled["label_stage1"] = (
            labeled["label_gold"].eq(1)
            | labeled["is_exact_complementary_pair_in_draw"].eq(1)
            | ((labeled["best_partner_union_coverage"] >= 0.95) & (labeled["best_partner_jaccard"] <= 0.10))
        ).astype(int)
        labeled = self._apply_analyst_labels(labeled)
        labeled["sample_weight"] = 1.0
        labeled.loc[labeled["label_tier"].eq("gold_analyst"), "sample_weight"] = 1.0
        return labeled

    def _apply_analyst_labels(self, labeled: pd.DataFrame) -> pd.DataFrame:
        try:
            from fraud_detection.utils.mongo_predictions import get_analyst_labels_collection

            docs = list(get_analyst_labels_collection().find({}, {"_id": 0}))
        except Exception as exc:
            logger.info("Analyst labels unavailable during feature engineering: %s", exc)
            return labeled
        if not docs:
            return labeled

        latest = pd.DataFrame(docs)
        required = {"draw_id", "member_id", "label"}
        if not required.issubset(latest.columns):
            return labeled
        latest["member_id"] = latest["member_id"].astype(str).str.strip().str.upper()
        latest["draw_id"] = pd.to_numeric(latest["draw_id"], errors="coerce").astype("Int64")
        if "decided_at" in latest.columns:
            latest["decided_at"] = pd.to_datetime(latest["decided_at"], errors="coerce", utc=True)
            latest = latest.sort_values("decided_at")
        latest = latest.dropna(subset=["draw_id"]).drop_duplicates(["member_id", "draw_id"], keep="last")
        decisions = {
            (row.member_id, row.draw_id): row.label
            for row in latest[["member_id", "draw_id", "label"]].itertuples(index=False)
        }
        draw_ids = pd.to_numeric(labeled["draw_id"], errors="coerce").astype("Int64")
        keys = list(zip(labeled["member_id"].astype(str).str.upper(), draw_ids))
        for idx, key in zip(labeled.index, keys):
            decision = decisions.get(key)
            if decision == "fraud":
                labeled.at[idx, "label_gold"] = 1
                labeled.at[idx, "label_stage1"] = 1
                labeled.at[idx, "label_source"] = "analyst"
                labeled.at[idx, "label_tier"] = "gold_analyst"
            elif decision == "not_fraud":
                labeled.at[idx, "label_gold"] = 0
                labeled.at[idx, "label_stage1"] = 0
                labeled.at[idx, "label_source"] = "analyst"
                labeled.at[idx, "label_tier"] = "gold_analyst"
        return labeled
