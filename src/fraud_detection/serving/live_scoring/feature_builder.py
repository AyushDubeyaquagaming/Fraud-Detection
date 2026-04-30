from __future__ import annotations

import pandas as pd

from fraud_detection.components.feature_engineering import compute_single_player_features


class FeatureBuilder:
    """Build a single-member feature vector using shared training-compatible logic."""

    def __init__(self, feature_columns: list[str], ccs_stats_lookup):
        self.feature_columns = feature_columns
        self.ccs_stats_lookup = ccs_stats_lookup

    def build(self, member_id: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame()

        features = compute_single_player_features(
            raw_df=raw_df,
            ccs_stats_lookup=self.ccs_stats_lookup,
        )
        if features.empty:
            return pd.DataFrame()

        features["member_id"] = str(member_id).strip().upper()
        if "primary_ccs_id" not in features.columns:
            features["primary_ccs_id"] = None

        missing = [col for col in self.feature_columns if col not in features.columns]
        if missing:
            raise ValueError(
                "Single-player feature computation did not produce the expected columns: "
                f"{missing}"
            )

        return features[["member_id"] + self.feature_columns + ["primary_ccs_id"]]
