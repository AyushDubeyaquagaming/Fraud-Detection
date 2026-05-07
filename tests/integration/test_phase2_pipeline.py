from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.components.feature_engineering import FeatureEngineering
from fraud_detection.components.model_training import ModelTraining
from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS, PairRuleConfig
from fraud_detection.components.partnership_features import STAGE1_FEATURE_COLUMNS
from fraud_detection.components.partnership_features import compute_partnership_features_from_candidates
from fraud_detection.entity.artifact_entity import DataIngestionArtifact, FeatureEngineeringArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig, ModelTrainingConfig


def _candidate_row(draw_id: int = 1) -> dict:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    return {
        "draw_id": draw_id,
        "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
        "trans_date_max": datetime(2026, 4, 27, 0, 1, tzinfo=timezone.utc),
        "qualifying_player_count": 2,
        "member_ids": [f"A{draw_id}", f"B{draw_id}"],
        "ccs_ids": ["CA", "CB"],
        "total_bet_amounts": [1000.0, 1000.0],
        "win_points": [1200.0, 1000.0],
        "coverage_bytes": [bytes(left), bytes(right)],
        "amount_vector": [[10.0 if v else 0.0 for v in left], [10.0 if v else 0.0 for v in right]],
    }


def test_candidate_partition_to_pair_and_member_projection(tmp_path) -> None:
    store = tmp_path / "candidate_draws" / "year=2026" / "month=04" / "week=18"
    store.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_candidate_row()]), store / "draws.parquet")
    candidate_df = pd.read_parquet(store / "draws.parquet")

    stage1_df, pair_df, _ = compute_partnership_features_from_candidates(
        candidate_df,
        rule_config=PairRuleConfig(),
        mode="training",
        ordinary_negative_sample=0,
        rolling_context=False,
    )

    assert not pair_df.empty
    assert not stage1_df.empty
    assert int(pair_df["is_strict_match"].sum()) == 1


def test_stage1_model_training_on_synthetic_pair_data(tmp_path) -> None:
    labels = pd.DataFrame(
        [
            {
                "draw_id": 1,
                "draw_date": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "member_a": "A",
                "member_b": "B",
                **{column: 1.0 for column in PAIR_FEATURE_COLUMNS},
                "is_strict_match": 1,
                "is_nearmiss": 0,
                "sampled_negative": 0,
                "label_stage1": 1,
                "label_gold": 0,
                "sample_weight": 1.0,
            },
            {
                "draw_id": 2,
                "draw_date": datetime(2026, 4, 28, tzinfo=timezone.utc),
                "member_a": "C",
                "member_b": "D",
                **{column: 0.0 for column in PAIR_FEATURE_COLUMNS},
                "is_strict_match": 0,
                "is_nearmiss": 1,
                "sampled_negative": 0,
                "label_stage1": 0,
                "label_gold": 0,
                "sample_weight": 1.0,
            },
        ]
    )
    labels_path = tmp_path / "stage1_labels.parquet"
    pair_path = tmp_path / "pair_events.parquet"
    stage2_path = tmp_path / "stage2_features.parquet"
    labels.to_parquet(labels_path, index=False)
    labels.to_parquet(pair_path, index=False)
    pd.DataFrame().to_parquet(stage2_path, index=False)
    fe_artifact = FeatureEngineeringArtifact(
        player_features_path=stage2_path,
        history_df_path=tmp_path,
        fraud_player_count=0,
        dropped_positive_count=0,
        feature_columns=PAIR_FEATURE_COLUMNS,
        feature_summary_path=tmp_path / "summary.json",
        mode="training_eval",
        stage1_labels_path=labels_path,
        stage2_features_path=stage2_path,
        partnership_table_path=tmp_path / "partnership.parquet",
        pair_events_path=pair_path,
    )

    artifact = ModelTraining(
        ModelTrainingConfig(random_seed=42, output_dir=tmp_path / "model_training", partnership={"use_candidate_store": True}),
        fe_artifact,
    ).initiate_model_training()

    assert artifact.model_bundle_path.exists()
    assert artifact.stage1_oof_predictions_path.exists()


def test_feature_engineering_candidate_store_smoke(tmp_path) -> None:
    store = tmp_path / "candidate_draws" / "year=2026" / "month=04" / "week=18"
    store.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_candidate_row(1), _candidate_row(2)]), store / "draws.parquet")
    ingestion = DataIngestionArtifact(
        raw_data_path=tmp_path / "candidate_draws",
        ingestion_report_path=tmp_path / "ingestion.json",
        row_count=2,
        member_count=4,
        source_type="candidate_store",
    )
    config = FeatureEngineeringConfig(
        fraud_csv_path=tmp_path / "fraud.csv",
        output_dir=tmp_path / "fe",
        mode="training_eval",
        partnership={
            "use_candidate_store": True,
            "candidate_store_path": str(tmp_path / "candidate_draws"),
            "candidate_window": {"start_date": "2026-04-27", "end_date": "2026-04-28"},
            "emit_negatives_sample": 0,
        },
    )

    artifact = FeatureEngineering(config, ingestion).initiate_feature_engineering()

    assert artifact.stage1_labels_path.exists()
    assert pd.read_parquet(artifact.stage1_labels_path).shape[0] == 2
    assert artifact.player_features_path == artifact.stage1_features_path
    assert artifact.feature_columns == STAGE1_FEATURE_COLUMNS
