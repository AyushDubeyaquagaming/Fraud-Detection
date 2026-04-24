from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.components.model_training import make_model_frame


@dataclass
class LiveScoreResult:
    member_id: str
    risk_score: float
    risk_tier: str
    anomaly_score: float
    supervised_score: float
    iso_forest_score: float
    mahalanobis_score: float
    cluster_distance_score: float
    ccs_id: str | None


class LiveScorer:
    """Apply the promoted hybrid bundle to a single-row feature frame."""

    def __init__(self, model_bundle: dict):
        self.bundle = model_bundle
        self._validate_bundle()

    def _validate_bundle(self) -> None:
        required = [
            "iso_forest",
            "kmeans",
            "scaler_unsup",
            "scaler_operational",
            "lr_operational",
            "feature_columns",
            "log1p_columns",
            "anomaly_weight",
            "supervised_weight",
            "anomaly_component_weights",
            "iso_min",
            "iso_max",
            "mahal_min",
            "mahal_max",
            "cluster_min",
            "cluster_max",
            "risk_p80",
            "risk_p95",
        ]
        for key in required:
            if key not in self.bundle:
                raise KeyError(f"Model bundle missing required key: {key}")
        if "mahal_stats" not in self.bundle and ("mean_vec" not in self.bundle or "cov_inv" not in self.bundle):
            raise KeyError("Model bundle missing Mahalanobis stats.")

    def _mahal_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if "mahal_stats" in self.bundle:
            stats = self.bundle["mahal_stats"]
            return stats["mean_vec"], stats["cov_inv"]
        return self.bundle["mean_vec"], self.bundle["cov_inv"]

    @staticmethod
    def _normalize_component(value: float, min_v: float, max_v: float) -> float:
        if max_v <= min_v:
            return 0.0
        return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.5))

    def score(self, feature_df: pd.DataFrame) -> LiveScoreResult:
        if len(feature_df) == 0:
            raise ValueError("Cannot score empty feature DataFrame")
        if len(feature_df) > 1:
            raise ValueError(f"Expected single-row feature DataFrame, got {len(feature_df)}")

        feature_cols = self.bundle["feature_columns"]
        model_frame = make_model_frame(
            feature_df,
            self.bundle["log1p_columns"],
            feature_cols,
        )

        X_unsup = self.bundle["scaler_unsup"].transform(model_frame)
        X_operational = self.bundle["scaler_operational"].transform(model_frame)

        iso_raw = float(-self.bundle["iso_forest"].score_samples(X_unsup)[0])
        mean_vec, cov_inv = self._mahal_stats()
        mahal_raw = float(np.sqrt(np.dot(np.dot((X_unsup[0] - mean_vec), cov_inv), (X_unsup[0] - mean_vec).T)))
        cluster_id = int(self.bundle["kmeans"].predict(X_unsup)[0])
        cluster_raw = float(np.linalg.norm(X_unsup[0] - self.bundle["kmeans"].cluster_centers_[cluster_id]))

        iso_norm = self._normalize_component(iso_raw, self.bundle["iso_min"], self.bundle["iso_max"])
        mahal_norm = self._normalize_component(mahal_raw, self.bundle["mahal_min"], self.bundle["mahal_max"])
        cluster_norm = self._normalize_component(cluster_raw, self.bundle["cluster_min"], self.bundle["cluster_max"])

        component_weights = self.bundle["anomaly_component_weights"]
        anomaly_score = (
            float(component_weights["iso_forest_score_norm"]) * iso_norm
            + float(component_weights["mahalanobis_norm"]) * mahal_norm
            + float(component_weights["cluster_distance_norm"]) * cluster_norm
        )
        supervised_score = float(self.bundle["lr_operational"].predict_proba(X_operational)[0, 1])
        risk_score = (
            float(self.bundle["anomaly_weight"]) * anomaly_score
            + float(self.bundle["supervised_weight"]) * supervised_score
        )

        if risk_score <= float(self.bundle["risk_p80"]):
            risk_tier = "LOW"
        elif risk_score <= float(self.bundle["risk_p95"]):
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"

        ccs_id = feature_df.iloc[0].get("primary_ccs_id")
        return LiveScoreResult(
            member_id=str(feature_df.iloc[0]["member_id"]),
            risk_score=float(risk_score),
            risk_tier=risk_tier,
            anomaly_score=float(anomaly_score),
            supervised_score=supervised_score,
            iso_forest_score=iso_raw,
            mahalanobis_score=mahal_raw,
            cluster_distance_score=cluster_raw,
            ccs_id=str(ccs_id) if pd.notna(ccs_id) else None,
        )
