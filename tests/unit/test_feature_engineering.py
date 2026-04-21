from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection.components.feature_engineering import (
    FeatureEngineering,
    gini_coeff,
    make_bet_template,
    safe_entropy,
)


# --- safe_entropy tests ---

def test_entropy_uniform():
    result = safe_entropy([1, 1, 1, 1])
    assert abs(result - math.log2(4)) < 1e-9


def test_entropy_concentrated():
    result = safe_entropy([10, 0, 0, 0])
    assert result == 0.0


def test_entropy_zeros():
    assert safe_entropy([0, 0, 0]) == 0.0


# --- gini_coeff tests ---

def test_gini_uniform():
    result = gini_coeff([5, 5, 5, 5])
    assert abs(result) < 1e-9


def test_gini_empty():
    assert gini_coeff([]) == 0.0


def test_gini_positive():
    result = gini_coeff([1, 10, 100])
    assert 0.0 <= result <= 1.0


# --- make_bet_template tests ---

def test_template_order_invariant():
    bets_a = [{"number": "5", "bet_amount": 10}, {"number": "3", "bet_amount": 20}]
    bets_b = [{"number": "3", "bet_amount": 20}, {"number": "5", "bet_amount": 10}]
    assert make_bet_template(bets_a) == make_bet_template(bets_b)


def test_template_excludes_zero_bets():
    bets = [{"number": "1", "bet_amount": 0}, {"number": "2", "bet_amount": 5}]
    template = make_bet_template(bets)
    assert len(template) == 1
    assert template[0][0] == "2"


def test_template_empty_list():
    assert make_bet_template([]) == tuple()


def test_template_non_list():
    assert make_bet_template(None) == tuple()


# --- Pre-fraud cutoff integration test ---

def test_pre_fraud_cutoff_keeps_only_pre_fraud_rows():
    """Player with fraud at draw 100 → only draws < 100 retained."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Build synthetic raw data: player A001 plays draws 50-150
        rows = []
        for draw in range(50, 151):
            rows.append({
                "member_id": "A001",
                "draw_id": draw,
                "total_bet_amount": 10.0,
                "win_points": 5.0,
                "bets": '[{"number": "1", "bet_amount": 10}]',
                "session_id": 1,
                "ccs_id": "CCS1",
                "createdAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw),
            })
        raw_df = pd.DataFrame(rows)
        raw_path = tmpdir / "raw_data.parquet"
        raw_df.to_parquet(raw_path, index=False)

        # Fraud CSV says A001 committed fraud at draw 100
        fraud_csv_path = tmpdir / "fraud.csv"
        pd.DataFrame({"member_id": ["A001"], "draw_id": [100]}).to_csv(fraud_csv_path, index=False)

        config = FeatureEngineeringConfig(
            exclude_cols=["member_id", "event_fraud_flag", "primary_ccs_id",
                          "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player"],
            log1p_cols=[],
            apply_pre_fraud_cutoff=True,
            fraud_csv_path=fraud_csv_path,
            output_dir=tmpdir / "fe_output",
            mode="training_eval",
        )
        ingestion_artifact = DataIngestionArtifact(
            raw_data_path=raw_path,
            ingestion_report_path=raw_path.parent / "ingestion_report.json",
            row_count=len(raw_df),
            member_count=1,
            source_type="parquet",
        )
        artifact = FeatureEngineering(config, ingestion_artifact).initiate_feature_engineering()

        # Check history_df only has pre-fraud draws
        history_df = pd.read_parquet(artifact.history_df_path)
        history_for_a001 = history_df[history_df["member_id"] == "A001"]

        # draw_id is Int64, cast for comparison
        draws = set(int(d) for d in history_for_a001["draw_id"].dropna())
        assert all(d < 100 for d in draws), f"Expected only draws < 100, got: {sorted(draws)}"


def test_fraud_at_draw_zero_dropped():
    """Player with fraud at draw 0 (first draw) → no pre-fraud history → dropped."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        rows = []
        for draw in range(0, 5):
            rows.append({
                "member_id": "B001",
                "draw_id": draw,
                "total_bet_amount": 10.0,
                "win_points": 5.0,
                "bets": '[{"number": "1", "bet_amount": 10}]',
                "session_id": 1,
                "ccs_id": "CCS2",
                "createdAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw),
            })
        raw_df = pd.DataFrame(rows)
        raw_path = tmpdir / "raw_data.parquet"
        raw_df.to_parquet(raw_path, index=False)

        fraud_csv_path = tmpdir / "fraud.csv"
        pd.DataFrame({"member_id": ["B001"], "draw_id": [0]}).to_csv(fraud_csv_path, index=False)

        config = FeatureEngineeringConfig(
            exclude_cols=["member_id", "event_fraud_flag", "primary_ccs_id",
                          "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player"],
            log1p_cols=[],
            apply_pre_fraud_cutoff=True,
            fraud_csv_path=fraud_csv_path,
            output_dir=tmpdir / "fe_output",
            mode="training_eval",
        )
        ingestion_artifact = DataIngestionArtifact(
            raw_data_path=raw_path,
            ingestion_report_path=raw_path.parent / "ingestion_report.json",
            row_count=len(raw_df),
            member_count=1,
            source_type="parquet",
        )
        artifact = FeatureEngineering(config, ingestion_artifact).initiate_feature_engineering()

        player_features = pd.read_parquet(artifact.player_features_path)
        # B001 had fraud at draw 0, no pre-fraud draws → should not appear in player features
        assert "B001" not in player_features["member_id"].values


def _build_synthetic_feature_inputs(tmp_path):
    """Build a small synthetic parquet + fraud CSV for parity / edge-case tests.

    Cohort shape:
      - 6 non-fraud members across 2 ccs_ids with varying history lengths
      - 2 fraud members with pre-fraud history (should survive cutoff)
      - 1 fraud member whose first draw IS the fraud event (no pre-fraud → dropped)
      - 1 fraud member with TWO fraud events where earliest timestamp and
        lowest draw_id are on DIFFERENT rows (exercises first-fraud selection)
    """
    import json
    from pathlib import Path

    rows = []
    fraud_rows = []
    base_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    draw_id = 1

    def _row(member, draw, bet, ccs, ts, session=0):
        return {
            "member_id": member,
            "draw_id": draw,
            "bets": json.dumps([{"number": str(draw % 38), "bet_amount": bet}]),
            "win_points": bet * 0.9,
            "total_bet_amount": bet,
            "session_id": session,
            "ccs_id": ccs,
            "createdAt": ts,
            "updatedAt": ts,
            "trans_date": ts,
        }

    # Non-fraud cohort: 6 members, variable history lengths
    for member_idx in range(6):
        member = f"N{member_idx:03d}"
        ccs = f"CCS_{member_idx % 2}"
        history_len = 5 + member_idx * 3
        for draw_idx in range(history_len):
            rows.append(_row(
                member=member,
                draw=draw_id,
                bet=5.0 + draw_idx * 0.1,
                ccs=ccs,
                ts=base_ts + pd.Timedelta(minutes=draw_id),
                session=draw_idx // 5,
            ))
            draw_id += 1

    # Fraud with pre-fraud history (two members)
    for member_idx in range(2):
        member = f"F{member_idx:03d}"
        ccs = "CCS_F"
        pre_fraud_count = 8
        for draw_idx in range(pre_fraud_count):
            rows.append(_row(
                member=member,
                draw=draw_id,
                bet=4.0,
                ccs=ccs,
                ts=base_ts + pd.Timedelta(minutes=draw_id),
            ))
            draw_id += 1
        # Fraud event
        fraud_draw_id = draw_id
        rows.append(_row(
            member=member,
            draw=fraud_draw_id,
            bet=50.0,
            ccs=ccs,
            ts=base_ts + pd.Timedelta(minutes=draw_id),
        ))
        fraud_rows.append({"member_id": member, "draw_id": fraud_draw_id})
        draw_id += 1
        # One post-fraud row (should be dropped by cutoff)
        rows.append(_row(
            member=member,
            draw=draw_id,
            bet=45.0,
            ccs=ccs,
            ts=base_ts + pd.Timedelta(minutes=draw_id),
        ))
        draw_id += 1

    # Fraud with NO pre-fraud history: first draw IS the fraud event
    no_hist_member = "FN01"
    no_hist_fraud_draw = draw_id
    rows.append(_row(
        member=no_hist_member,
        draw=no_hist_fraud_draw,
        bet=100.0,
        ccs="CCS_F",
        ts=base_ts + pd.Timedelta(minutes=draw_id),
    ))
    fraud_rows.append({"member_id": no_hist_member, "draw_id": no_hist_fraud_draw})
    draw_id += 1

    # Fraud with TWO fraud events whose (ts, draw_id) minima are on different rows.
    # Earliest timestamp is on draw 9000 (large draw_id).
    # Lowest draw_id is 100 but its timestamp is LATER.
    # → first_fraud_ts and first_fraud_draw_id must both come from draw 9000.
    multi_member = "FM01"
    # Pre-fraud history before the earliest fraud
    for pre in range(3):
        rows.append(_row(
            member=multi_member,
            draw=draw_id,
            bet=3.0,
            ccs="CCS_F",
            ts=pd.Timestamp("2024-02-01T00:00:00Z") + pd.Timedelta(minutes=pre),
        ))
        draw_id += 1
    # Fraud event A — EARLIER ts, LARGER draw_id
    fraud_a_ts = pd.Timestamp("2024-02-10T00:00:00Z")
    fraud_a_draw = 9000
    rows.append(_row(
        member=multi_member,
        draw=fraud_a_draw,
        bet=60.0,
        ccs="CCS_F",
        ts=fraud_a_ts,
    ))
    fraud_rows.append({"member_id": multi_member, "draw_id": fraud_a_draw})
    # Fraud event B — LATER ts, SMALLER draw_id
    fraud_b_ts = pd.Timestamp("2024-02-20T00:00:00Z")
    fraud_b_draw = 100
    rows.append(_row(
        member=multi_member,
        draw=fraud_b_draw,
        bet=65.0,
        ccs="CCS_F",
        ts=fraud_b_ts,
    ))
    fraud_rows.append({"member_id": multi_member, "draw_id": fraud_b_draw})
    # Some post-fraud-B row
    rows.append(_row(
        member=multi_member,
        draw=draw_id + 100,
        bet=10.0,
        ccs="CCS_F",
        ts=fraud_b_ts + pd.Timedelta(minutes=1),
    ))

    raw_df = pd.DataFrame(rows)
    fraud_df = pd.DataFrame(fraud_rows)
    raw_path = tmp_path / "raw_data.parquet"
    fraud_path = tmp_path / "fraud.csv"
    raw_df.to_parquet(raw_path, index=False)
    fraud_df.to_csv(fraud_path, index=False)
    return raw_path, fraud_path, raw_df, fraud_df


def _run_fe_with_mode(tmp_path, raw_path, fraud_path, force_mode: str, out_subdir: str):
    from pathlib import Path
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact

    config = FeatureEngineeringConfig(
        exclude_cols=[
            "member_id", "event_fraud_flag", "primary_ccs_id",
            "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player",
        ],
        log1p_cols=[],
        apply_pre_fraud_cutoff=True,
        fraud_csv_path=fraud_path,
        output_dir=tmp_path / out_subdir,
        mode="training_eval",
    )
    raw_rows = pd.read_parquet(raw_path).shape[0]
    ingestion_artifact = DataIngestionArtifact(
        raw_data_path=raw_path,
        ingestion_report_path=raw_path.parent / "ingestion_report.json",
        row_count=raw_rows,
        member_count=pd.read_parquet(raw_path)["member_id"].nunique(),
        source_type="parquet",
    )
    return FeatureEngineering(
        config, ingestion_artifact, _force_mode=force_mode
    ).initiate_feature_engineering()


def test_bucketed_path_matches_in_memory_path_on_small_data(tmp_path):
    """Both feature engineering paths must produce identical player-level outputs
    on the same synthetic input. The bucketed path is only safe if it matches
    the simpler in-memory path row-for-row.
    """
    raw_path, fraud_path, _, _ = _build_synthetic_feature_inputs(tmp_path)

    artifact_in_memory = _run_fe_with_mode(tmp_path, raw_path, fraud_path, "in_memory", "fe_in_memory")
    artifact_bucketed = _run_fe_with_mode(tmp_path, raw_path, fraud_path, "bucketed", "fe_bucketed")

    pf_in_memory = pd.read_parquet(artifact_in_memory.player_features_path)
    pf_bucketed = pd.read_parquet(artifact_bucketed.player_features_path)

    assert set(pf_in_memory.columns) == set(pf_bucketed.columns), (
        f"Column mismatch: only in in_memory={set(pf_in_memory.columns) - set(pf_bucketed.columns)}, "
        f"only in bucketed={set(pf_bucketed.columns) - set(pf_in_memory.columns)}"
    )
    shared_cols = sorted(pf_in_memory.columns)

    pf_in_memory = pf_in_memory.sort_values("member_id").reset_index(drop=True)[shared_cols]
    pf_bucketed = pf_bucketed.sort_values("member_id").reset_index(drop=True)[shared_cols]

    assert len(pf_in_memory) == len(pf_bucketed)
    assert pf_in_memory["member_id"].tolist() == pf_bucketed["member_id"].tolist()

    for col in shared_cols:
        a = pf_in_memory[col]
        b = pf_bucketed[col]
        assert str(a.dtype) == str(b.dtype), (
            f"Dtype mismatch for {col}: in_memory={a.dtype} vs bucketed={b.dtype}"
        )
        if pd.api.types.is_numeric_dtype(a):
            # Int64 nullable columns may contain pd.NA; convert both to float with NA→NaN
            if str(a.dtype) in {"Int64", "Int32", "Int16", "Int8"}:
                a = a.astype("Float64")
                b = b.astype("Float64")
            a_arr = np.where(pd.isna(a), np.nan, a.astype(float))
            b_arr = np.where(pd.isna(b), np.nan, b.astype(float))
            np.testing.assert_allclose(
                a_arr, b_arr, rtol=1e-9, atol=1e-12, equal_nan=True,
                err_msg=f"Numeric value mismatch for column {col}",
            )
        else:
            assert a.fillna("<NA>").tolist() == b.fillna("<NA>").tolist(), (
                f"Non-numeric value mismatch for column {col}"
            )

    assert artifact_in_memory.fraud_player_count == artifact_bucketed.fraud_player_count
    assert artifact_in_memory.dropped_positive_count == artifact_bucketed.dropped_positive_count


def test_bucketed_hashing_keeps_each_member_in_exactly_one_bucket(tmp_path):
    """The bucketed path is only correct if every member lands in exactly one
    bucket (so per-bucket aggregation yields one row per member).
    Verify by computing the hash directly and checking single-bucket mapping.
    """
    from fraud_detection.components.feature_engineering import FeatureEngineering

    raw_path, fraud_path, raw_df, _ = _build_synthetic_feature_inputs(tmp_path)

    bucket_count = 8
    hashed = (
        pd.util.hash_pandas_object(raw_df["member_id"].astype(str), index=False)
        .astype("uint64")
        .to_numpy()
        % bucket_count
    )
    per_member = pd.DataFrame({"member_id": raw_df["member_id"].astype(str), "bucket": hashed})
    unique_buckets_per_member = per_member.groupby("member_id")["bucket"].nunique()
    assert (unique_buckets_per_member == 1).all(), (
        f"Some members land in >1 bucket: "
        f"{unique_buckets_per_member[unique_buckets_per_member > 1].to_dict()}"
    )

    # Confirm the end-to-end bucketed path still yields exactly one row per member
    artifact = _run_fe_with_mode(tmp_path, raw_path, fraud_path, "bucketed", "fe_bucketed_unique")
    pf = pd.read_parquet(artifact.player_features_path)
    assert pf["member_id"].is_unique, "player_features has duplicate member_id after bucketed aggregation"


def test_first_fraud_event_selection_uses_one_consistent_row(tmp_path):
    """For a member with multiple fraud events where min(ts) and min(draw_id)
    fall on DIFFERENT rows, first_fraud_ts and first_fraud_draw_id must both
    come from the same earliest-by-(ts,draw_id) event row.

    Regression guard for the bug where `.agg({'ts': 'min', 'draw_id': 'min'})`
    selects field values independently.
    """
    raw_path, fraud_path, _, _ = _build_synthetic_feature_inputs(tmp_path)
    artifact = _run_fe_with_mode(tmp_path, raw_path, fraud_path, "in_memory", "fe_multi_fraud")
    history_df = pd.read_parquet(artifact.history_df_path)

    multi = history_df[history_df["member_id"] == "FM01"]
    assert not multi.empty, "Multi-fraud member FM01 missing from history_df"

    # first_fraud_ts must equal the earlier of the two fraud events (2024-02-10)
    # and first_fraud_draw_id must equal that SAME row's draw_id (9000),
    # NOT the global min draw_id across fraud events (100).
    ffts_values = multi["first_fraud_ts"].dropna().unique()
    ffdi_values = multi["first_fraud_draw_id"].dropna().unique()
    assert len(ffts_values) == 1
    assert len(ffdi_values) == 1

    assert pd.Timestamp(ffts_values[0]).tz_convert("UTC") == pd.Timestamp("2024-02-10T00:00:00Z")
    assert int(ffdi_values[0]) == 9000, (
        f"first_fraud_draw_id should be 9000 (same row as earliest ts), got {int(ffdi_values[0])}"
    )


def test_bucketed_handles_empty_bucket(tmp_path):
    """If bucketing assigns zero rows to a given bucket (possible with very
    skewed hash input on small cohorts), the path must skip it cleanly and
    still produce valid player_features / history outputs.
    """
    from pathlib import Path
    import json
    # Tiny cohort: one member → with 8 buckets, seven will be empty.
    rows = []
    for draw_idx in range(20):
        rows.append({
            "member_id": "SINGLE",
            "draw_id": draw_idx,
            "bets": json.dumps([{"number": "1", "bet_amount": 10}]),
            "win_points": 5.0,
            "total_bet_amount": 10.0,
            "session_id": 0,
            "ccs_id": "CCS1",
            "createdAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw_idx),
            "updatedAt": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw_idx),
            "trans_date": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=draw_idx),
        })
    raw_df = pd.DataFrame(rows)
    raw_path = tmp_path / "raw_data.parquet"
    raw_df.to_parquet(raw_path, index=False)
    fraud_path = tmp_path / "fraud.csv"
    pd.DataFrame({"member_id": [], "draw_id": []}).to_csv(fraud_path, index=False)

    artifact = _run_fe_with_mode(tmp_path, raw_path, fraud_path, "bucketed", "fe_empty_bucket")
    pf = pd.read_parquet(artifact.player_features_path)
    assert len(pf) == 1
    assert pf.iloc[0]["member_id"] == "SINGLE"


def test_event_key_construction_matches_between_fraud_csv_and_parquet(tmp_path):
    """Regression guard: both the fraud-CSV loader and the parquet normalizer must
    produce IDENTICAL event keys for the same (member_id, draw_id) pair.

    Historical bugs in similar pipelines include: casting draw_id from float
    (produces '.0' suffix), inconsistent casing of member_id, and using a
    different separator character between the two code paths.
    """
    from fraud_detection.components.feature_engineering import FeatureEngineering
    from fraud_detection.entity.config_entity import FeatureEngineeringConfig
    from fraud_detection.entity.artifact_entity import DataIngestionArtifact
    import json

    # Same (member_id, draw_id) expressed with mixed-case header in the CSV
    # and whitespace padding — emulates the real ROULET CHEATING DATA.csv format.
    fraud_csv_path = tmp_path / "fraud.csv"
    fraud_csv_path.write_text(
        "DATE,DRAW_ID,MEMBER_ID,CCS_ID\n"
        "12/18/2025,7102365, gk00206537 ,4778\n"
        "12/18/2025,7102366,GK00206537,4778\n"
    )

    # Parquet row with matching (normalized) key
    raw_df = pd.DataFrame([
        {
            "member_id": "gk00206537",  # lowercase — normalizer must upper it
            "draw_id": 7102365,
            "bets": json.dumps([{"number": "1", "bet_amount": 10}]),
            "win_points": 5.0,
            "total_bet_amount": 10.0,
            "session_id": 0,
            "ccs_id": "4778",
            "createdAt": pd.Timestamp("2025-12-17T00:00:00Z"),
            "updatedAt": pd.Timestamp("2025-12-17T00:00:00Z"),
            "trans_date": pd.Timestamp("2025-12-17T00:00:00Z"),
        },
        {
            "member_id": "GK00206537",
            "draw_id": 7102300,  # pre-fraud row
            "bets": json.dumps([{"number": "1", "bet_amount": 8}]),
            "win_points": 4.0,
            "total_bet_amount": 8.0,
            "session_id": 0,
            "ccs_id": "4778",
            "createdAt": pd.Timestamp("2025-12-16T00:00:00Z"),
            "updatedAt": pd.Timestamp("2025-12-16T00:00:00Z"),
            "trans_date": pd.Timestamp("2025-12-16T00:00:00Z"),
        },
    ])
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)

    config = FeatureEngineeringConfig(
        exclude_cols=[
            "member_id", "event_fraud_flag", "primary_ccs_id",
            "first_fraud_ts", "first_fraud_draw_id", "is_fraud_player",
        ],
        log1p_cols=[],
        apply_pre_fraud_cutoff=True,
        fraud_csv_path=fraud_csv_path,
        output_dir=tmp_path / "fe_out",
        mode="training_eval",
    )
    ingestion_artifact = DataIngestionArtifact(
        raw_data_path=raw_path,
        ingestion_report_path=raw_path.parent / "ingestion_report.json",
        row_count=len(raw_df),
        member_count=1,
        source_type="parquet",
    )
    artifact = FeatureEngineering(config, ingestion_artifact).initiate_feature_engineering()
    pf = pd.read_parquet(artifact.player_features_path)
    # Member should be recognized as fraud despite mixed case / whitespace in CSV
    assert artifact.fraud_player_count == 1
    assert pf["event_fraud_flag"].sum() == 1
    # Pre-fraud row should survive the cutoff
    history = pd.read_parquet(artifact.history_df_path)
    assert len(history[history["member_id"] == "GK00206537"]) >= 1


def test_history_schema_stable_across_null_and_typed_fraud_columns(tmp_path):
    history_path = tmp_path / "history_df.parquet"
    writers = {}

    base_columns = {
        "member_id": ["A001"],
        "draw_id": pd.Series([1], dtype="Int64"),
        "bets": ['[{"number": "1", "bet_amount": 10}]'],
        "win_points": [5.0],
        "total_bet_amount": [10.0],
        "session_id": [1],
        "ccs_id": ["CCS1"],
        "createdAt": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "updatedAt": [pd.Timestamp("2024-01-01T00:00:30Z")],
        "trans_date": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "ts": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "bets_per_draw": [1],
        "nonzero_bets_per_draw": [1],
        "tiny_bet_ratio_in_draw": [0.0],
        "max_bet_share_in_draw": [1.0],
        "bet_amount_std_in_draw": [0.0],
        "bet_amount_mean_in_draw": [10.0],
        "entropy_in_draw": [0.0],
        "gini_in_draw": [0.0],
        "unique_positions_in_draw": [1],
        "position_coverage": [1 / 38.0],
        "net_result": [-5.0],
        "bet_template": ['[["1",10.0]]'],
        "fraud_event_key": ["1|A001"],
        "inter_draw_seconds": [0.0],
        "event_label": [0],
        "is_fraud_player": [0],
    }

    null_df = FeatureEngineering._ensure_history_schema(pd.DataFrame(base_columns))

    typed_df = pd.DataFrame(
        {
            **base_columns,
            "member_id": ["A002"],
            "draw_id": pd.Series([2], dtype="Int64"),
            "fraud_event_key": ["2|A002"],
            "event_label": [1],
            "first_fraud_ts": [pd.Timestamp("2024-01-01T00:01:00Z")],
            "first_fraud_draw_id": pd.Series([2], dtype="Int64"),
            "is_fraud_player": [1],
        }
    )
    typed_df = FeatureEngineering._ensure_history_schema(typed_df)

    FeatureEngineering._append_dataframe_to_parquet(null_df, history_path, writers)
    FeatureEngineering._append_dataframe_to_parquet(typed_df, history_path, writers)
    FeatureEngineering._close_writers(writers)

    history_df = pd.read_parquet(history_path)
    assert len(history_df) == 2
    assert str(history_df["first_fraud_ts"].dtype).startswith("datetime64")
    assert str(history_df["first_fraud_draw_id"].dtype) == "Int64"
