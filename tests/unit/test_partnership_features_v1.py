from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.components.feature_engineering import FeatureEngineering
from fraud_detection.components.partnership_features import (
    ROULETTE_POSITIONS,
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    build_stage2_training_frame,
    compute_partnership_features,
)
from fraud_detection.components.partnership_modeling import train_stage1_oof, train_stage2_model
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import FeatureEngineeringConfig


def _bets(positions: list[str], amount: float = 100.0) -> str:
    return json.dumps([{"number": position, "bet_amount": amount} for position in positions])


def _raw_exact_draw() -> pd.DataFrame:
    left = ROULETTE_POSITIONS[:19]
    right = ROULETTE_POSITIONS[19:]
    return pd.DataFrame(
        [
            {
                "draw_id": 10,
                "member_id": "a",
                "bets": _bets(left),
                "total_bet_amount": 1900.0,
                "win_points": 2500.0,
                "trans_date": pd.Timestamp("2026-04-21T10:00:00Z"),
            },
            {
                "draw_id": 10,
                "member_id": "b",
                "bets": _bets(right),
                "total_bet_amount": 1900.0,
                "win_points": 2500.0,
                "trans_date": pd.Timestamp("2026-04-21T10:00:01Z"),
            },
        ]
    )


def _candidate_row(draw_id: int) -> dict:
    left = [1] * 19 + [0] * 19
    right = [0] * 19 + [1] * 19
    return {
        "draw_id": draw_id,
        "trans_date_min": pd.Timestamp("2026-04-27T00:00:00Z"),
        "trans_date_max": pd.Timestamp("2026-04-27T00:00:01Z"),
        "qualifying_player_count": 2,
        "member_ids": [f"A{draw_id}", f"B{draw_id}"],
        "ccs_ids": ["CA", "CB"],
        "total_bet_amounts": [1900.0, 1900.0],
        "win_points": [2500.0, 2500.0],
        "coverage_bytes": [bytes(left), bytes(right)],
        "amount_vector": [
            [100.0 if value else 0.0 for value in left],
            [100.0 if value else 0.0 for value in right],
        ],
    }


def test_compute_partnership_features_detects_exact_complementary_draw():
    stage1, pair_events, _ = compute_partnership_features(_raw_exact_draw())

    assert set(stage1["member_id"]) == {"A", "B"}
    assert set(STAGE1_FEATURE_COLUMNS).issubset(stage1.columns)
    assert stage1["is_exact_complementary_pair_in_draw"].tolist() == [1, 1]
    assert len(pair_events) == 1
    assert int(pair_events.iloc[0]["union_positions"]) == 38
    assert int(pair_events.iloc[0]["overlap_positions"]) == 0


def test_unprofitable_partition_is_not_section_b_exact():
    raw = _raw_exact_draw()
    raw["win_points"] = 100.0

    stage1, pair_events, _ = compute_partnership_features(raw)

    assert len(pair_events) == 1
    assert int(pair_events.iloc[0]["is_exact"]) == 0
    assert stage1["is_exact_complementary_pair_in_draw"].sum() == 0


def test_candidate_only_stage1_excludes_non_candidates():
    raw = pd.concat(
        [
            _raw_exact_draw(),
            pd.DataFrame(
                [
                    {
                        "draw_id": 10,
                        "member_id": "c",
                        "bets": _bets(["1"], amount=10.0),
                        "total_bet_amount": 10.0,
                        "win_points": 0.0,
                        "trans_date": pd.Timestamp("2026-04-21T10:00:02Z"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    stage1, _, _ = compute_partnership_features(raw, candidate_only=True)

    assert set(stage1["member_id"]) == {"A", "B"}


def test_stage2_training_frame_excludes_padding_columns():
    preds = pd.DataFrame(
        [
            {"member_id": "A", "draw_id": 1, "draw_date": pd.Timestamp("2026-04-21"), "stage1_score": 0.9, "best_partner_member_id": "B"},
            {"member_id": "A", "draw_id": 2, "draw_date": pd.Timestamp("2026-04-22"), "stage1_score": 0.1, "best_partner_member_id": None},
            {"member_id": "C", "draw_id": 3, "draw_date": pd.Timestamp("2026-04-22"), "stage1_score": 0.0, "best_partner_member_id": None},
        ]
    )

    stage2 = build_stage2_training_frame(preds, gold_members={"A"})

    assert "avg_tiny_bet_ratio" not in stage2.columns
    assert "template_reuse_ratio" not in stage2.columns
    assert dict(zip(stage2["member_id"], stage2["label_gold_member"])) == {"A": 1, "C": 0}


def test_train_stage1_oof_marks_oof_rows():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "member_id": f"M{idx}",
                "draw_id": idx,
                "draw_date": pd.Timestamp("2026-04-21"),
                "label_stage1": int(idx % 3 == 0),
                "sample_weight": 1.0,
                **{col: float(idx % 5) for col in STAGE1_FEATURE_COLUMNS},
            }
        )
    frame = pd.DataFrame(rows)

    result = train_stage1_oof(frame, n_splits=2)

    assert result.metrics["oof_rows"] == len(frame)
    assert result.predictions["is_oof"].all()
    assert result.predictions["stage1_score"].between(0, 1).all()


def test_train_stage2_model_returns_oof_predictions_and_final_model():
    rows = []
    for idx in range(20):
        rows.append(
            {
                "member_id": f"M{idx}",
                "label_gold_member": int(idx % 5 == 0),
                **{col: float(idx % 7) for col in STAGE2_FEATURE_COLUMNS},
            }
        )
    frame = pd.DataFrame(rows)

    result = train_stage2_model(frame, n_splits=2)

    assert result.metrics["validation_status"] == "evaluated_oof"
    assert result.metrics["oof_rows"] == len(frame)
    assert result.predictions["is_oof"].all()
    assert set(result.predictions["stage2_prediction_source"]) == {"oof"}
    assert result.predictions["stage2_score"].between(0, 1).all()
    assert result.model.predict_proba(frame[result.feature_columns]).shape[0] == len(frame)


def test_train_stage2_model_marks_insufficient_oof_without_using_in_sample_scores():
    frame = pd.DataFrame(
        [
            {"member_id": "A", "label_gold_member": 1, **{col: 1.0 for col in STAGE2_FEATURE_COLUMNS}},
            {"member_id": "B", "label_gold_member": 0, **{col: 0.0 for col in STAGE2_FEATURE_COLUMNS}},
        ]
    )

    result = train_stage2_model(frame, n_splits=5)

    assert result.metrics["validation_status"] == "insufficient_labels_for_oof"
    assert result.metrics["oof_rows"] == 0
    assert not result.predictions["is_oof"].any()
    assert set(result.predictions["stage2_prediction_source"]) == {"insufficient_labels_for_oof"}


def test_feature_engineering_streams_candidate_only_stage1_rows(tmp_path: Path):
    raw = pd.concat(
        [
            _raw_exact_draw(),
            pd.DataFrame(
                [
                    {
                        "draw_id": 10,
                        "member_id": "c",
                        "bets": _bets(["1"], amount=10.0),
                        "total_bet_amount": 10.0,
                        "win_points": 0.0,
                        "trans_date": pd.Timestamp("2026-04-21T10:00:02Z"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, index=False)
    fraud_path = tmp_path / "fraud.csv"
    pd.DataFrame({"member_id": ["A"], "draw_id": [10]}).to_csv(fraud_path, index=False)

    config = FeatureEngineeringConfig(
        fraud_csv_path=fraud_path,
        output_dir=tmp_path / "fe",
        mode="training_eval",
        partnership={"stream_batch_size": 2},
    )
    artifact = FeatureEngineering(
        config,
        DataIngestionArtifact(
            raw_data_path=raw_path,
            ingestion_report_path=tmp_path / "ingestion.json",
            row_count=len(raw),
            member_count=3,
            source_type="parquet",
        ),
    ).initiate_feature_engineering()

    stage1_labeled = pd.read_parquet(artifact.stage1_labels_path)

    assert set(stage1_labeled["member_id"]) == {"A", "B"}


def test_candidate_store_feature_engineering_disables_analyst_overlay_after_first_failure(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "candidate_draws" / "year=2026" / "month=04" / "week=18"
    store.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_candidate_row(1), _candidate_row(2)]), store / "draws.parquet")

    calls = {"count": 0}

    def fail_read_analyst_labels_for_draws(draw_ids=None):
        calls["count"] += 1
        raise RuntimeError("analyst labels backend unavailable")

    monkeypatch.setattr(
        "fraud_detection.utils.mongo_predictions.read_analyst_labels_for_draws",
        fail_read_analyst_labels_for_draws,
    )

    artifact = FeatureEngineering(
        FeatureEngineeringConfig(
            fraud_csv_path=tmp_path / "fraud.csv",
            output_dir=tmp_path / "fe_candidate",
            mode="training_eval",
            partnership={
                "use_candidate_store": True,
                "candidate_store_path": str(tmp_path / "candidate_draws"),
                "candidate_window": {"start_date": "2026-04-27", "end_date": "2026-04-28"},
                "emit_negatives_sample": 0,
                "stream_batch_size": 1,
                "analyst_label_overlay": {"enabled": True},
            },
        ),
        DataIngestionArtifact(
            raw_data_path=tmp_path / "candidate_draws",
            ingestion_report_path=tmp_path / "ingestion.json",
            row_count=2,
            member_count=4,
            source_type="candidate_store",
        ),
    ).initiate_feature_engineering()

    assert artifact.stage1_labels_path.exists()
    assert pd.read_parquet(artifact.stage1_labels_path).shape[0] == 2
    assert calls["count"] == 1
