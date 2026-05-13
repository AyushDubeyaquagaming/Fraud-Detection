from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.components.ccs_features import CCS_FEATURE_COLUMNS
from fraud_detection.components.feature_engineering import FeatureEngineering
from fraud_detection.components.model_training import ModelTraining
from fraud_detection.components.pair_scan import PAIR_FEATURE_COLUMNS
from fraud_detection.components.partnership_features import STAGE2_FEATURE_COLUMNS
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig, ModelTrainingConfig
from fraud_detection.serving.live_scoring.draw_scorer import DrawScorer
from fraud_detection.utils.common import load_joblib


def _candidate(draw_id: int, ccs_a: str = "CA", ccs_b: str = "CB") -> dict:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    return {
        "draw_id": draw_id,
        "trans_date_min": datetime(2026, 5, 7, tzinfo=timezone.utc),
        "trans_date_max": datetime(2026, 5, 7, 0, 1, tzinfo=timezone.utc),
        "qualifying_player_count": 2,
        "member_ids": [f"A{draw_id}", f"B{draw_id}"],
        "ccs_ids": [ccs_a, ccs_b],
        "total_bet_amounts": [1000.0, 1000.0],
        "win_points": [1200.0, 1000.0],
        "coverage_bytes": [bytes(left), bytes(right)],
        "amount_vector": [[10.0 if value else 0.0 for value in left], [10.0 if value else 0.0 for value in right]],
    }


def test_phase3_training_contract_includes_cross_ccs_and_ccs_features(tmp_path):
    store = tmp_path / "candidate_draws" / "year=2026" / "month=05" / "week=19"
    store.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_candidate(1), _candidate(2)]), store / "draws.parquet")
    profit_store = tmp_path / "ccs_daily_profit" / "year=2026" / "month=05"
    profit_store.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ccs_id": "CA", "member_id": "A1", "profit_date": datetime(2026, 5, 7).date(), "daily_profit": 80.0},
            {"ccs_id": "CA", "member_id": "X", "profit_date": datetime(2026, 5, 7).date(), "daily_profit": 20.0},
            {"ccs_id": "CB", "member_id": "B1", "profit_date": datetime(2026, 5, 7).date(), "daily_profit": 10.0},
        ]
    ).to_parquet(profit_store / "ccs_daily_profit.parquet", index=False)

    ingestion = DataIngestionArtifact(
        raw_data_path=tmp_path / "candidate_draws",
        ingestion_report_path=tmp_path / "ingestion.json",
        row_count=2,
        member_count=4,
        source_type="candidate_store",
    )
    partnership = {
        "use_candidate_store": True,
        "candidate_store_path": str(tmp_path / "candidate_draws"),
        "candidate_window": {"start_date": "2026-05-07", "end_date": "2026-05-08"},
        "emit_negatives_sample": 0,
        "ccs_features": {
            "enabled": True,
            "profit_path": str(tmp_path / "ccs_daily_profit"),
            "windows_days": [1, 7],
            "concentration_threshold": 0.70,
        },
    }
    fe_artifact = FeatureEngineering(
        FeatureEngineeringConfig(
            fraud_csv_path=tmp_path / "fraud.csv",
            output_dir=tmp_path / "fe",
            mode="training_eval",
            partnership=partnership,
        ),
        ingestion,
    ).initiate_feature_engineering()
    training_artifact = ModelTraining(
        ModelTrainingConfig(random_seed=42, output_dir=tmp_path / "model_training", partnership=partnership),
        fe_artifact,
    ).initiate_model_training()

    assert "pair_different_ccs" in PAIR_FEATURE_COLUMNS
    assert "ccs_profit_share_7d" in STAGE2_FEATURE_COLUMNS
    assert training_artifact.model_bundle_path.exists()
    assert training_artifact.ccs_concentration_table_path.exists()
    ccs_table = pd.read_parquet(training_artifact.ccs_concentration_table_path)
    assert {"ccs_id", "member_id", "profit_date", "daily_profit"}.issubset(ccs_table.columns)

    scorer = DrawScorer(
        load_joblib(training_artifact.model_bundle_path),
        ccs_concentration_table=ccs_table,
    )
    stage1_rows = pd.DataFrame(
        {
            "member_id": ["A1"],
            "ccs_id": ["CA"],
            "draw_id": [999],
            "draw_date": [pd.Timestamp("2026-05-07", tz="UTC")],
            "best_partner_member_id": ["B1"],
            "stage1_score": [0.9],
        }
    )
    attached = scorer._attach_serving_ccs_context(stage1_rows)
    assert set(CCS_FEATURE_COLUMNS).issubset(attached.columns)
    assert attached["ccs_profit_share_1d"].iloc[0] > 0.0

    result = scorer.score_candidate_draw(_candidate(1))
    assert result.flagged_members
    assert result.flagged_members[0]["ccs_profit_share_1d"] > 0.0
