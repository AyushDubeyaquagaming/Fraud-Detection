from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.extraction.candidate_extractor import (
    CandidateDrawExtractor,
    CandidateExtractionConfig,
    ExtractionConfig,
    MongoConfig,
    OutputConfig,
    RuntimeConfig,
    _build_position_index,
    _player_to_vectors,
    parse_utc_date,
    split_weekly_chunks,
    transform_grouped_doc,
)
from fraud_detection.extraction.parquet_writer import read_partition_chunk_bounds, write_candidate_rows
from fraud_detection.extraction.pipeline_builder import build_candidate_pipeline


BOARD_POSITIONS = ["0", "00", *[str(number) for number in range(1, 37)]]


def _config() -> ExtractionConfig:
    return ExtractionConfig(
        game="roulette",
        min_total_bet_amount=1000.0,
        min_qualifying_players=2,
        board_size=38,
        board_positions=BOARD_POSITIONS,
        timestamp_field="trans_date",
    )


def test_player_to_vectors_builds_coverage_and_amounts() -> None:
    player = {
        "bets": [
            {"number": "0", "bet_amount": 25},
            {"number": "00", "bet_amount": 50},
            {"number": 7, "bet_amount": "100.5"},
            {"number": 8, "bet_amount": 0},
            {"number": "outside", "bet_amount": 75},
        ]
    }

    coverage, amounts = _player_to_vectors(player, _build_position_index(BOARD_POSITIONS), 38)

    assert len(coverage) == 38
    assert len(amounts) == 38
    assert coverage[0] == 1
    assert coverage[1] == 1
    assert coverage[8] == 1
    assert coverage[9] == 0
    assert amounts[0] == 25.0
    assert amounts[1] == 50.0
    assert amounts[8] == 100.5
    assert amounts[9] == 0.0
    assert sum(coverage) == 3


def test_transform_grouped_doc_preserves_aligned_player_arrays() -> None:
    doc = {
        "_id": 12345,
        "trans_date_min": datetime(2026, 4, 25, 10, 0, tzinfo=timezone.utc),
        "trans_date_max": datetime(2026, 4, 25, 10, 1, tzinfo=timezone.utc),
        "qualifying_player_count": 3,
        "players": [
            {
                "member_id": "M1",
                "ccs_id": "C1",
                "total_bet_amount": 1200,
                "win_points": 100,
                "bets": [{"number": "0", "bet_amount": 100}, {"number": 1, "bet_amount": 200}],
            },
            {
                "member_id": "M2",
                "ccs_id": "C2",
                "total_bet_amount": 1500,
                "win_points": None,
                "bets": [{"number": "00", "bet_amount": 300}, {"number": 36, "bet_amount": 400}],
            },
            {
                "member_id": "M3",
                "ccs_id": None,
                "total_bet_amount": "1800.5",
                "win_points": "250.5",
                "bets": [{"number": 7, "bet_amount": 50}],
            },
        ],
    }

    row = transform_grouped_doc(
        doc,
        _config(),
        extraction_run_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )

    assert row["draw_id"] == 12345
    assert row["qualifying_player_count"] == 3
    assert row["member_ids"] == ["M1", "M2", "M3"]
    assert row["ccs_ids"] == ["C1", "C2", None]
    assert row["total_bet_amounts"] == [1200.0, 1500.0, 1800.5]
    assert row["win_points"] == [100.0, 0.0, 250.5]
    assert len(row["member_ids"]) == len(row["coverage_bytes"]) == len(row["amount_vector"])

    m1_index = row["member_ids"].index("M1")
    assert row["coverage_bytes"][m1_index][0] == 1
    assert row["coverage_bytes"][m1_index][2] == 1
    assert row["amount_vector"][m1_index][0] == 100.0
    assert row["amount_vector"][m1_index][2] == 200.0

    m2_index = row["member_ids"].index("M2")
    assert row["coverage_bytes"][m2_index][1] == 1
    assert row["coverage_bytes"][m2_index][37] == 1
    assert row["amount_vector"][m2_index][1] == 300.0
    assert row["amount_vector"][m2_index][37] == 400.0

    m3_index = row["member_ids"].index("M3")
    assert row["coverage_bytes"][m3_index][8] == 1
    assert row["amount_vector"][m3_index][8] == 50.0


def test_pipeline_uses_match_padding_but_assigns_by_chunk_min_timestamp() -> None:
    chunk_start = datetime(2026, 4, 27, tzinfo=timezone.utc)
    chunk_end = datetime(2026, 5, 4, tzinfo=timezone.utc)
    match_start = datetime(2026, 4, 26, 23, 55, tzinfo=timezone.utc)
    match_end = datetime(2026, 5, 4, 0, 5, tzinfo=timezone.utc)

    pipeline = build_candidate_pipeline(
        chunk_start,
        chunk_end,
        "roulette",
        1000.0,
        2,
        timestamp_field="createdAt",
        match_start=match_start,
        match_end=match_end,
    )

    assert pipeline[0]["$match"]["createdAt"] == {"$gte": match_start, "$lt": match_end}
    assert pipeline[1]["$group"]["trans_date_min"] == {"$min": "$createdAt"}
    assert pipeline[2]["$match"]["trans_date_min"] == {"$gte": chunk_start, "$lt": chunk_end}


def test_split_weekly_chunks_handles_monday_non_midnight_start() -> None:
    chunks = split_weekly_chunks(
        datetime(2026, 4, 27, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 27, 13, 0, tzinfo=timezone.utc),
        Path("data_store/candidate_draws"),
    )

    assert [(chunk.chunk_start.isoformat(), chunk.chunk_end.isoformat()) for chunk in chunks] == [
        ("2026-04-27T09:00:00+00:00", "2026-04-27T13:00:00+00:00")
    ]


def test_player_to_vectors_accepts_json_string_bets() -> None:
    player = {"bets": '[{"number": "1", "bet_amount": 10}]'}

    coverage, amounts = _player_to_vectors(player, _build_position_index(BOARD_POSITIONS), 38)

    assert coverage[2] == 1
    assert amounts[2] == 10.0


def test_parse_utc_date_accepts_z_suffix() -> None:
    parsed = parse_utc_date("2026-04-25T00:00:00Z")

    assert parsed.isoformat() == "2026-04-25T00:00:00+00:00"


def test_write_candidate_rows_stores_chunk_metadata_and_replaces_target(tmp_path: Path) -> None:
    output_path = tmp_path / "year=2026" / "month=04" / "week=18" / "draws.parquet"
    rows = [
        {
            "draw_id": 1,
            "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
            "trans_date_max": datetime(2026, 4, 27, 0, 1, tzinfo=timezone.utc),
            "qualifying_player_count": 2,
            "member_ids": ["A", "B"],
            "ccs_ids": ["C1", "C2"],
            "total_bet_amounts": [1000.0, 1200.0],
            "win_points": [0.0, 0.0],
            "coverage_bytes": [b"\x01" * 38, b"\x00" * 38],
            "amount_vector": [[1.0] * 38, [0.0] * 38],
            "extraction_run_at": datetime(2026, 5, 5, tzinfo=timezone.utc),
        }
    ]

    write_candidate_rows(
        rows,
        output_path,
        chunk_start=datetime(2026, 4, 27, tzinfo=timezone.utc),
        chunk_end=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )

    assert output_path.exists()
    assert read_partition_chunk_bounds(output_path) == (
        "2026-04-27T00:00:00+00:00",
        "2026-05-04T00:00:00+00:00",
    )
    assert not list(output_path.parent.glob("*.tmp"))


def test_existing_partition_without_metadata_is_reextracted(tmp_path: Path) -> None:
    output_path = tmp_path / "data_store" / "candidate_draws" / "year=2026" / "month=04" / "week=18" / "draws.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_table = pa.Table.from_pylist(
        [
            {
                "draw_id": 9,
                "trans_date_min": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "trans_date_max": datetime(2026, 4, 27, tzinfo=timezone.utc),
                "qualifying_player_count": 2,
                "member_ids": ["OLD"],
                "ccs_ids": ["OLD"],
                "total_bet_amounts": [1000.0],
                "win_points": [0.0],
                "coverage_bytes": [b"\x00" * 38],
                "amount_vector": [[0.0] * 38],
                "extraction_run_at": datetime(2026, 5, 5, tzinfo=timezone.utc),
            }
        ]
    )
    pq.write_table(legacy_table, output_path)

    extractor = CandidateDrawExtractor(
        CandidateExtractionConfig(
            mongo=MongoConfig("A", "B", "C"),
            extraction=_config(),
            output=OutputConfig(
                base_path=tmp_path / "data_store" / "candidate_draws",
                partition_strategy="weekly_iso",
                parquet_compression="zstd",
                parquet_row_group_size=50000,
            ),
            runtime=RuntimeConfig(
                log_level="INFO",
                resume_existing_partitions=True,
                fail_fast_on_chunk_error=False,
            ),
        )
    )

    class _Collection:
        def aggregate(self, *args, **kwargs):
            return iter(
                [
                    {
                        "_id": 123,
                        "trans_date_min": datetime(2026, 4, 27, 0, 30, tzinfo=timezone.utc),
                        "trans_date_max": datetime(2026, 4, 27, 0, 31, tzinfo=timezone.utc),
                        "qualifying_player_count": 2,
                        "players": [
                            {
                                "member_id": "M1",
                                "ccs_id": "C1",
                                "total_bet_amount": 1000.0,
                                "win_points": 0.0,
                                "bets": [{"number": "1", "bet_amount": 10}],
                            },
                            {
                                "member_id": "M2",
                                "ccs_id": "C2",
                                "total_bet_amount": 1200.0,
                                "win_points": 0.0,
                                "bets": [{"number": "2", "bet_amount": 20}],
                            },
                        ],
                    }
                ]
            )

    partition = split_weekly_chunks(
        datetime(2026, 4, 27, tzinfo=timezone.utc),
        datetime(2026, 5, 4, tzinfo=timezone.utc),
        tmp_path / "data_store" / "candidate_draws",
    )[0]

    result = extractor._run_partition(
        partition,
        collection=_Collection(),
        is_first_chunk=True,
        is_last_chunk=True,
        force=False,
        dry_run=False,
    )

    assert result.status == "succeeded"
    assert read_partition_chunk_bounds(output_path) == (
        "2026-04-27T00:00:00+00:00",
        "2026-05-04T00:00:00+00:00",
    )
