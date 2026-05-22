from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from fraud_detection.components.ccs_features import (
    CCS_FEATURE_COLUMNS,
    attach_ccs_concentration_features,
    load_relevant_profit_rows,
)
from fraud_detection.components.partnership_features import (
    PAIR_STAGE1_FEATURE_COLUMNS,
    SECTION_A_FEATURE_COLUMNS,
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    PartnershipThresholds,
    build_stage2_training_frame,
    project_pair_scores_to_member_draw_rows,
)
from fraud_detection.components.partnership_modeling import train_stage1_oof, train_stage2_model
from fraud_detection.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainingArtifact
from fraud_detection.entity.config_entity import ModelTrainingConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import get_logger
from fraud_detection.utils.common import ensure_dir, save_joblib, write_json

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _native_feedback_member_labels(
    pair_scored: pd.DataFrame, partnership_cfg: dict[str, Any]
) -> tuple[set[str], dict[str, float]]:
    native_cfg = partnership_cfg.get("native_feedback", {}) or {}
    fraud_weight = float(native_cfg.get("confirmed_fraud_weight", 2.0))
    not_fraud_weight = float(native_cfg.get("confirmed_not_fraud_weight", 2.0))
    fraud_members: set[str] = set()
    member_weights: dict[str, float] = {}
    for member_col, label_col, weight_col in [
        ("member_a", "native_member_a_label", "native_member_a_weight"),
        ("member_b", "native_member_b_label", "native_member_b_weight"),
    ]:
        if member_col not in pair_scored.columns or label_col not in pair_scored.columns:
            continue
        weights = (
            pd.to_numeric(pair_scored.get(weight_col), errors="coerce").fillna(1.0)
            if weight_col in pair_scored.columns
            else pd.Series(1.0, index=pair_scored.index)
        )
        for member_id, label, weight in zip(pair_scored[member_col], pair_scored[label_col], weights):
            member_key = str(member_id or "").strip().upper()
            label_value = str(label or "").strip().lower()
            if not member_key or label_value not in {"fraud", "not_fraud"}:
                continue
            if label_value == "fraud":
                fraud_members.add(member_key)
                effective_weight = max(float(weight), fraud_weight)
            else:
                effective_weight = max(float(weight), not_fraud_weight)
            member_weights[member_key] = max(member_weights.get(member_key, 1.0), effective_weight)
    return fraud_members, member_weights


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, fe_artifact: FeatureEngineeringArtifact):
        self.config = config
        self.fe_artifact = fe_artifact

    def initiate_model_training(self) -> ModelTrainingArtifact:
        logger.info("PartnershipModelTraining: starting")
        try:
            ensure_dir(self.config.output_dir)
            if self.fe_artifact.stage1_labels_path is None:
                raise ValueError("FeatureEngineeringArtifact.stage1_labels_path is required.")
            stage1_labeled = pd.read_parquet(self.fe_artifact.stage1_labels_path)
            partnership_features = (
                pd.read_parquet(self.fe_artifact.partnership_table_path)
                if self.fe_artifact.partnership_table_path and self.fe_artifact.partnership_table_path.exists()
                else pd.DataFrame()
            )

            use_candidate_store = bool(self.config.partnership.get("use_candidate_store", False))
            stage1_feature_columns = PAIR_STAGE1_FEATURE_COLUMNS if use_candidate_store else STAGE1_FEATURE_COLUMNS
            stage1_result = train_stage1_oof(
                stage1_labeled,
                feature_columns=stage1_feature_columns,
                random_seed=self.config.random_seed,
            )
            stage1_oof_path = self.config.output_dir / "stage1_oof_predictions.parquet"
            stage1_result.predictions.to_parquet(stage1_oof_path, index=False)

            if use_candidate_store:
                pair_scored = stage1_labeled.merge(
                    stage1_result.predictions[
                        [
                            column
                            for column in ["draw_id", "member_a", "member_b", "stage1_score"]
                            if column in stage1_result.predictions.columns
                        ]
                    ],
                    on=["draw_id", "member_a", "member_b"],
                    how="left",
                )
                strict_signal = (
                    pd.to_numeric(
                        pair_scored.get("is_strict_collusion_pattern", pair_scored.get("is_strict_match", 0)),
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                )
                pair_scored["pair_risk_score"] = strict_signal.where(
                    strict_signal.eq(1),
                    pd.to_numeric(pair_scored["stage1_score"], errors="coerce").fillna(0.0),
                )
                nearmiss_values = pair_scored.get("is_nearmiss", pd.Series(0, index=pair_scored.index))
                nearmiss_mask = pd.to_numeric(nearmiss_values, errors="coerce").fillna(0).astype(int).eq(1)
                pair_scored.loc[nearmiss_mask, "pair_risk_score"] = pair_scored.loc[
                    nearmiss_mask, "pair_risk_score"
                ].clip(lower=float(self.config.partnership.get("stage1_flag_threshold", 0.70)))
                stage1_for_stage2 = project_pair_scores_to_member_draw_rows(pair_scored, rolling_context=True)
                gold_members = set()
                if "label_gold" in pair_scored.columns:
                    gold_pairs = pair_scored.loc[pair_scored["label_gold"].eq(1)]
                    gold_members = set(gold_pairs["member_a"].astype(str).str.upper()) | set(
                        gold_pairs["member_b"].astype(str).str.upper()
                    )
                native_fraud_members, native_member_weights = _native_feedback_member_labels(
                    pair_scored,
                    self.config.partnership,
                )
                gold_members |= native_fraud_members
                score_rows = []
                for row in pair_scored.to_dict("records"):
                    score = float(row.get("pair_risk_score") or 0.0)
                    for member_col in ("member_a", "member_b"):
                        score_rows.append(
                            {
                                "member_id": str(row.get(member_col, "")).strip().upper(),
                                "draw_id": int(row.get("draw_id")),
                                "stage1_score": score,
                            }
                        )
                member_scores = (
                    pd.DataFrame(score_rows).groupby(["member_id", "draw_id"], as_index=False)["stage1_score"].max()
                    if score_rows
                    else pd.DataFrame(columns=["member_id", "draw_id", "stage1_score"])
                )
                stage1_predictions_for_stage2 = stage1_for_stage2.merge(
                    member_scores,
                    on=["member_id", "draw_id"],
                    how="left",
                )
                stage1_predictions_for_stage2["stage1_score"] = pd.to_numeric(
                    stage1_predictions_for_stage2["stage1_score"], errors="coerce"
                ).fillna(0.0)
                stage1_signal_features = (
                    pd.read_parquet(self.fe_artifact.stage1_features_path)
                    if self.fe_artifact.stage1_features_path and self.fe_artifact.stage1_features_path.exists()
                    else pd.DataFrame()
                )
                if not stage1_signal_features.empty and "stage1_score" in stage1_signal_features.columns:
                    clique_signal_rows = stage1_signal_features.loc[
                        pd.to_numeric(stage1_signal_features["stage1_score"], errors="coerce").fillna(0.0) > 0.0
                    ].copy()
                    if not clique_signal_rows.empty:
                        stage1_predictions_for_stage2 = pd.concat(
                            [stage1_predictions_for_stage2, clique_signal_rows],
                            ignore_index=True,
                        )
                        stage1_predictions_for_stage2 = stage1_predictions_for_stage2.sort_values(
                            ["member_id", "draw_id", "stage1_score", "best_partner_union_coverage"],
                            ascending=[True, True, False, False],
                        ).drop_duplicates(["member_id", "draw_id"], keep="first")
                ccs_cfg = self.config.partnership.get("ccs_features", {}) or {}
                if ccs_cfg.get("enabled", False):
                    stage1_predictions_for_stage2 = attach_ccs_concentration_features(
                        stage1_predictions_for_stage2,
                        ccs_profit_path=ccs_cfg.get("profit_path", "data_store/ccs_daily_profit"),
                        windows_days=list(ccs_cfg.get("windows_days", [1, 7])),
                        concentration_threshold=float(ccs_cfg.get("concentration_threshold", 0.70)),
                    )
            else:
                native_fraud_members = set()
                native_member_weights = {}
                gold_members = set(
                    stage1_labeled.loc[stage1_labeled["label_gold"].eq(1), "member_id"].astype(str).str.upper()
                )
                stage1_predictions_for_stage2 = stage1_result.predictions
            stage2_features = build_stage2_training_frame(
                stage1_predictions_for_stage2,
                partnership_features=partnership_features,
                gold_members=gold_members,
            )
            if native_member_weights and "member_id" in stage2_features.columns:
                stage2_features["sample_weight"] = (
                    stage2_features["member_id"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(native_member_weights)
                    .fillna(1.0)
                    .astype(float)
                )
            stage2_member_ids = (
                set(stage2_features["member_id"].astype(str).str.strip().str.upper())
                if "member_id" in stage2_features.columns
                else set()
            )
            native_fraud_members_in_stage2 = native_fraud_members & stage2_member_ids
            missing_native_fraud_members = sorted(native_fraud_members - stage2_member_ids)
            if missing_native_fraud_members:
                logger.warning(
                    "Stage 2 seeding: %d native feedback fraud member(s) did not reach the Stage 2 frame because they were not in candidate pairs.",
                    len(missing_native_fraud_members),
                )
            stage2_features_path = self.config.output_dir / "stage2_features.parquet"
            stage2_features.to_parquet(stage2_features_path, index=False)
            ccs_concentration_table_path = self.config.output_dir / "ccs_concentration_table.parquet"
            ccs_cfg = self.config.partnership.get("ccs_features", {}) or {}
            ccs_lookup = (
                load_relevant_profit_rows(
                    stage1_predictions_for_stage2,
                    ccs_profit_path=ccs_cfg.get("profit_path", "data_store/ccs_daily_profit"),
                    windows_days=list(ccs_cfg.get("windows_days", [1, 7])),
                )
                if ccs_cfg.get("enabled", False)
                else pd.DataFrame()
            )
            ccs_lookup.to_parquet(ccs_concentration_table_path, index=False)
            stage2_result = train_stage2_model(
                stage2_features,
                feature_columns=STAGE2_FEATURE_COLUMNS,
                random_seed=self.config.random_seed,
            )
            stage2_predictions_path = self.config.output_dir / "stage2_predictions.parquet"
            stage2_result.predictions.to_parquet(stage2_predictions_path, index=False)

            stage1_model_path = self.config.output_dir / "stage1_model.joblib"
            stage2_model_path = self.config.output_dir / "stage2_model.joblib"
            save_joblib(stage1_result.model, stage1_model_path)
            save_joblib(stage2_result.model, stage2_model_path)

            thresholds = PartnershipThresholds(**dict(self.config.partnership.get("candidate_thresholds", {})))
            pair_rules = dict(self.config.partnership.get("pair_rules", {}))
            if "strict_inference_filter" in self.config.partnership:
                pair_rules["strict_inference_filter"] = self.config.partnership["strict_inference_filter"]
            bundle = {
                "model_version": "partnership_v1",
                "stage1_model": stage1_result.model,
                "stage2_model": stage2_result.model,
                "stage1_feature_columns": stage1_feature_columns,
                "stage2_feature_columns": stage2_result.feature_columns,
                "use_candidate_store": use_candidate_store,
                "pair_rules": pair_rules,
                "clique_rules": dict(self.config.partnership.get("clique_rules", {})),
                "ccs_features": dict(self.config.partnership.get("ccs_features", {})),
                "section_a_feature_columns": SECTION_A_FEATURE_COLUMNS,
                "candidate_thresholds": thresholds.to_dict(),
                "stage1_high_threshold": float(self.config.partnership.get("stage1_high_threshold", 0.5)),
                "stage2_alert_threshold": float(self.config.partnership.get("stage2_alert_threshold", 0.65)),
                "rolling_window_days": int(self.config.partnership.get("rolling_window_days", 7)),
                "short_rolling_windows": list(self.config.partnership.get("short_rolling_windows", [1, 3])),
                "synthetic_positives_from_strict_pattern": bool(
                    self.config.partnership.get("synthetic_positives_from_strict_pattern", False)
                ),
                "label_tier_weights": self.config.partnership.get(
                    "label_tier_weights", {"gold": 1.0, "silver": 1.0, "bronze": 0.5, "gold_analyst": 1.0}
                ),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "training_window": {},
            }
            bundle_path = self.config.output_dir / "model_bundle.joblib"
            save_joblib(bundle, bundle_path)

            report = {
                "model_version": "partnership_v1",
                "stage1": stage1_result.metrics,
                "stage2": stage2_result.metrics,
                "stage1_rows": int(len(stage1_labeled)),
                "stage2_rows": int(len(stage2_features)),
                "fraud_members": (
                    int(stage2_features["label_gold_member"].sum()) if "label_gold_member" in stage2_features else 0
                ),
                "native_feedback_fraud_members": int(len(native_fraud_members)),
                "native_feedback_fraud_members_in_stage2": int(len(native_fraud_members_in_stage2)),
                "native_feedback_fraud_members_missing_stage2": int(len(missing_native_fraud_members)),
                "native_feedback_fraud_members_missing_stage2_sample": missing_native_fraud_members[:50],
                "native_feedback_weighted_members": int(len(native_member_weights)),
                "stage1_feature_columns": stage1_feature_columns,
                "stage2_feature_columns": stage2_result.feature_columns,
                "ccs_feature_columns": CCS_FEATURE_COLUMNS,
                "stage2_alert_threshold": bundle["stage2_alert_threshold"],
                "stage1_oof_predictions_path": str(stage1_oof_path),
                "stage2_predictions_path": str(stage2_predictions_path),
                "trained_at": bundle["trained_at"],
            }
            training_report_path = self.config.output_dir / "training_report.json"
            write_json(report, training_report_path)
            return ModelTrainingArtifact(
                training_report_path=training_report_path,
                feature_columns=stage2_result.feature_columns,
                stage1_model_path=stage1_model_path,
                stage2_model_path=stage2_model_path,
                model_bundle_path=bundle_path,
                stage1_oof_predictions_path=stage1_oof_path,
                stage2_features_path=stage2_features_path,
                partnership_table_path=self.fe_artifact.partnership_table_path,
                ccs_concentration_table_path=ccs_concentration_table_path,
                stage1_feature_columns=stage1_feature_columns,
                stage2_feature_columns=stage2_result.feature_columns,
            )
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc
