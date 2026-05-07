from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

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
                        [column for column in ["draw_id", "member_a", "member_b", "stage1_score"] if column in stage1_result.predictions.columns]
                    ],
                    on=["draw_id", "member_a", "member_b"],
                    how="left",
                )
                pair_scored["pair_risk_score"] = pair_scored["is_strict_match"].where(
                    pair_scored["is_strict_match"].eq(1),
                    pd.to_numeric(pair_scored["stage1_score"], errors="coerce").fillna(0.0),
                )
                stage1_for_stage2 = project_pair_scores_to_member_draw_rows(pair_scored, rolling_context=True)
                gold_members = set()
                if "label_gold" in pair_scored.columns:
                    gold_pairs = pair_scored.loc[pair_scored["label_gold"].eq(1)]
                    gold_members = set(gold_pairs["member_a"].astype(str).str.upper()) | set(
                        gold_pairs["member_b"].astype(str).str.upper()
                    )
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
            else:
                gold_members = set(stage1_labeled.loc[stage1_labeled["label_gold"].eq(1), "member_id"].astype(str).str.upper())
                stage1_predictions_for_stage2 = stage1_result.predictions
            stage2_features = build_stage2_training_frame(
                stage1_predictions_for_stage2,
                partnership_features=partnership_features,
                gold_members=gold_members,
            )
            stage2_features_path = self.config.output_dir / "stage2_features.parquet"
            stage2_features.to_parquet(stage2_features_path, index=False)
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
            bundle = {
                "model_version": "partnership_v1",
                "stage1_model": stage1_result.model,
                "stage2_model": stage2_result.model,
                "stage1_feature_columns": stage1_feature_columns,
                "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
                "use_candidate_store": use_candidate_store,
                "pair_rules": dict(self.config.partnership.get("pair_rules", {})),
                "section_a_feature_columns": SECTION_A_FEATURE_COLUMNS,
                "candidate_thresholds": thresholds.to_dict(),
                "stage1_high_threshold": float(self.config.partnership.get("stage1_high_threshold", 0.5)),
                "stage2_alert_threshold": float(self.config.partnership.get("stage2_alert_threshold", 0.65)),
                "rolling_window_days": int(self.config.partnership.get("rolling_window_days", 7)),
                "label_tier_weights": self.config.partnership.get("label_tier_weights", {"gold": 1.0, "silver": 1.0, "bronze": 0.5, "gold_analyst": 1.0}),
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
                "fraud_members": int(stage2_features["label_gold_member"].sum()) if "label_gold_member" in stage2_features else 0,
                "stage1_feature_columns": stage1_feature_columns,
                "stage2_feature_columns": STAGE2_FEATURE_COLUMNS,
                "stage2_alert_threshold": bundle["stage2_alert_threshold"],
                "stage1_oof_predictions_path": str(stage1_oof_path),
                "stage2_predictions_path": str(stage2_predictions_path),
                "trained_at": bundle["trained_at"],
            }
            training_report_path = self.config.output_dir / "training_report.json"
            write_json(report, training_report_path)
            return ModelTrainingArtifact(
                training_report_path=training_report_path,
                feature_columns=STAGE2_FEATURE_COLUMNS,
                stage1_model_path=stage1_model_path,
                stage2_model_path=stage2_model_path,
                model_bundle_path=bundle_path,
                stage1_oof_predictions_path=stage1_oof_path,
                stage2_features_path=stage2_features_path,
                partnership_table_path=self.fe_artifact.partnership_table_path,
                stage1_feature_columns=stage1_feature_columns,
                stage2_feature_columns=STAGE2_FEATURE_COLUMNS,
            )
        except Exception as exc:
            raise FraudDetectionException(exc, sys) from exc
