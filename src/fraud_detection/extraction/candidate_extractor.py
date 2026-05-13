from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from fraud_detection.exception import FraudDetectionException
from fraud_detection.extraction.parquet_writer import (
    PartitionInfo,
    read_partition_chunk_bounds,
    resolve_partition_path,
    write_candidate_rows,
)
from fraud_detection.extraction.pipeline_builder import build_candidate_pipeline
from fraud_detection.logger import get_logger
from fraud_detection.utils.player_vectors import build_position_index, player_to_vectors
from fraud_detection.utils.mongodb import get_mongo_collection

logger = get_logger(__name__)
BOUNDARY_PADDING = timedelta(minutes=5)


@dataclass(frozen=True)
class MongoConfig:
    uri_env_var: str
    database_env_var: str
    collection_env_var: str


@dataclass(frozen=True)
class ExtractionConfig:
    game: str
    min_total_bet_amount: float
    min_qualifying_players: int
    board_size: int
    board_positions: list[str]
    timestamp_field: str = "trans_date"


@dataclass(frozen=True)
class OutputConfig:
    base_path: Path
    partition_strategy: str
    parquet_compression: str
    parquet_row_group_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str
    resume_existing_partitions: bool
    fail_fast_on_chunk_error: bool


@dataclass(frozen=True)
class CandidateExtractionConfig:
    mongo: MongoConfig
    extraction: ExtractionConfig
    output: OutputConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class ChunkResult:
    status: str
    partition: PartitionInfo
    groups_returned: int = 0
    rows_written: int = 0
    qualifying_players: int = 0
    unknown_positions: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class ExtractionRunSummary:
    chunks_total: int
    succeeded: int
    skipped: int
    failed: int
    total_rows: int
    elapsed_seconds: float
    results: list[ChunkResult]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def load_candidate_extraction_config(config_path: str | Path) -> CandidateExtractionConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Candidate extraction config is empty or invalid: {path}")

    mongo = raw.get("mongo") or {}
    extraction = raw.get("extraction") or {}
    output = raw.get("output") or {}
    runtime = raw.get("runtime") or {}

    base_path = Path(output["base_path"])
    if not base_path.is_absolute():
        base_path = path.parent.parent / base_path

    return CandidateExtractionConfig(
        mongo=MongoConfig(
            uri_env_var=str(mongo["uri_env_var"]),
            database_env_var=str(mongo["database_env_var"]),
            collection_env_var=str(mongo["collection_env_var"]),
        ),
        extraction=ExtractionConfig(
            game=str(extraction["game"]),
            min_total_bet_amount=float(extraction["min_total_bet_amount"]),
            min_qualifying_players=int(extraction["min_qualifying_players"]),
            board_size=int(extraction["board_size"]),
            board_positions=[str(pos) for pos in extraction["board_positions"]],
            timestamp_field=str(extraction.get("timestamp_field", "trans_date")),
        ),
        output=OutputConfig(
            base_path=base_path,
            partition_strategy=str(output["partition_strategy"]),
            parquet_compression=str(output.get("parquet_compression", "zstd")),
            parquet_row_group_size=int(output.get("parquet_row_group_size", 50_000)),
        ),
        runtime=RuntimeConfig(
            log_level=str(runtime.get("log_level", "INFO")),
            resume_existing_partitions=bool(runtime.get("resume_existing_partitions", True)),
            fail_fast_on_chunk_error=bool(runtime.get("fail_fast_on_chunk_error", False)),
        ),
    )


def parse_utc_date(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD or ISO datetime.") from exc
    if parsed.tzinfo is None:
        return datetime.combine(parsed.date(), datetime_time.min, tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def split_weekly_chunks(start: datetime, end: datetime, base_path: Path) -> list[PartitionInfo]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes.")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if start >= end:
        raise ValueError("start date must be before end date.")

    chunks: list[PartitionInfo] = []
    current = start
    while current < end:
        days_until_next_monday = (7 - current.weekday()) % 7
        if days_until_next_monday == 0 and current.time() == datetime_time.min:
            next_boundary = current + timedelta(days=7)
        else:
            if days_until_next_monday == 0:
                days_until_next_monday = 7
            next_monday_date = (current + timedelta(days=days_until_next_monday)).date()
            next_boundary = datetime.combine(next_monday_date, datetime_time.min, tzinfo=timezone.utc)
        chunk_end = min(next_boundary, end)
        chunks.append(
            PartitionInfo(
                chunk_start=current,
                chunk_end=chunk_end,
                path=resolve_partition_path(base_path, current),
            )
        )
        current = chunk_end
    return chunks


def _build_position_index(board_positions: list[str]) -> dict[str, int]:
    return build_position_index(board_positions)


def _player_to_vectors(player: dict, position_index: dict, board_size: int) -> tuple[bytes, list[float]]:
    return _player_to_vectors_with_stats(player, position_index, board_size, None)


def _player_to_vectors_with_stats(
    player: dict,
    position_index: dict,
    board_size: int,
    unknown_positions: Counter[str] | None,
) -> tuple[bytes, list[float]]:
    return player_to_vectors(
        player,
        position_index,
        board_size,
        unknown_positions=unknown_positions,
    )


def _coerce_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def transform_grouped_doc(
    doc: dict,
    config: ExtractionConfig,
    *,
    extraction_run_at: datetime | None = None,
    unknown_positions: Counter[str] | None = None,
) -> dict:
    position_index = _build_position_index(config.board_positions)
    member_ids: list[str] = []
    ccs_ids: list[str | None] = []
    totals: list[float] = []
    wins: list[float] = []
    coverage_vectors: list[bytes] = []
    amount_vectors: list[list[float]] = []

    for player in doc.get("players") or []:
        coverage, amounts = _player_to_vectors_with_stats(
            player,
            position_index,
            config.board_size,
            unknown_positions,
        )
        member_ids.append(str(player.get("member_id", "")))
        ccs_value = player.get("ccs_id")
        ccs_ids.append(None if ccs_value is None else str(ccs_value))
        totals.append(float(player.get("total_bet_amount") or 0.0))
        wins.append(float(player.get("win_points") or 0.0))
        coverage_vectors.append(coverage)
        amount_vectors.append(amounts)

    return {
        "draw_id": int(doc["_id"]),
        "trans_date_min": _coerce_utc(doc["trans_date_min"]),
        "trans_date_max": _coerce_utc(doc["trans_date_max"]),
        "qualifying_player_count": int(doc["qualifying_player_count"]),
        "member_ids": member_ids,
        "ccs_ids": ccs_ids,
        "total_bet_amounts": totals,
        "win_points": wins,
        "coverage_bytes": coverage_vectors,
        "amount_vector": amount_vectors,
        "extraction_run_at": extraction_run_at or datetime.now(tz=timezone.utc),
    }


class CandidateDrawExtractor:
    def __init__(self, config: CandidateExtractionConfig):
        self.config = config
        if config.output.partition_strategy != "weekly_iso":
            raise ValueError(f"Unsupported partition strategy: {config.output.partition_strategy}")
        if len(config.extraction.board_positions) != config.extraction.board_size:
            raise ValueError("board_positions length must match board_size.")

    def run(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        force: bool = False,
        dry_run: bool = False,
        limit_chunks: int | None = None,
    ) -> ExtractionRunSummary:
        started = time.perf_counter()
        partitions = split_weekly_chunks(start_date, end_date, self.config.output.base_path)
        if limit_chunks is not None:
            partitions = partitions[:limit_chunks]
        results: list[ChunkResult] = []

        client = None
        collection = None
        if not dry_run:
            try:
                client, collection = get_mongo_collection(
                    self.config.mongo.uri_env_var,
                    self.config.mongo.database_env_var,
                    self.config.mongo.collection_env_var,
                )
            except Exception as exc:
                raise FraudDetectionException(exc, sys) from exc

        try:
            for index, partition in enumerate(partitions, start=1):
                result = self._run_partition(
                    partition,
                    collection=collection,
                    is_first_chunk=index == 1,
                    is_last_chunk=index == len(partitions),
                    force=force,
                    dry_run=dry_run,
                )
                results.append(result)
                if result.status == "failed" and self.config.runtime.fail_fast_on_chunk_error:
                    break
        finally:
            if client is not None:
                client.close()

        elapsed = time.perf_counter() - started
        summary = ExtractionRunSummary(
            chunks_total=len(partitions),
            succeeded=sum(1 for result in results if result.status == "succeeded"),
            skipped=sum(1 for result in results if result.status == "skipped"),
            failed=sum(1 for result in results if result.status == "failed"),
            total_rows=sum(result.rows_written for result in results),
            elapsed_seconds=elapsed,
            results=results,
        )
        logger.info(
            "[summary] chunks=%d succeeded=%d skipped=%d failed=%d total_rows=%d elapsed_total=%.2fs",
            summary.chunks_total,
            summary.succeeded,
            summary.skipped,
            summary.failed,
            summary.total_rows,
            summary.elapsed_seconds,
        )
        return summary

    def _run_partition(
        self,
        partition: PartitionInfo,
        *,
        collection: Any,
        is_first_chunk: bool,
        is_last_chunk: bool,
        force: bool,
        dry_run: bool,
    ) -> ChunkResult:
        started = time.perf_counter()
        if (
            partition.path.exists()
            and self.config.runtime.resume_existing_partitions
            and not force
        ):
            existing_start, existing_end = read_partition_chunk_bounds(partition.path)
            requested_start = partition.chunk_start.isoformat()
            requested_end = partition.chunk_end.isoformat()
            if (existing_start, existing_end) == (requested_start, requested_end):
                elapsed = time.perf_counter() - started
                logger.info(
                    "[chunk %s %s..%s] skipped (already exists) elapsed=%.2fs path=%s",
                    partition.label,
                    partition.chunk_start.date(),
                    partition.chunk_end.date(),
                    elapsed,
                    partition.path,
                )
                return ChunkResult(status="skipped", partition=partition, elapsed_seconds=elapsed)
            if (existing_start, existing_end) == (None, None):
                logger.warning(
                    "[chunk %s %s..%s] existing partition has no chunk metadata; re-extracting for safe resume path=%s",
                    partition.label,
                    partition.chunk_start.date(),
                    partition.chunk_end.date(),
                    partition.path,
                )
            else:
                logger.info(
                    "[chunk %s %s..%s] existing partition has different chunk bounds "
                    "(%s..%s), re-extracting path=%s",
                    partition.label,
                    partition.chunk_start.date(),
                    partition.chunk_end.date(),
                    existing_start,
                    existing_end,
                    partition.path,
                )

        if dry_run:
            elapsed = time.perf_counter() - started
            logger.info(
                "[chunk %s %s..%s] dry-run path=%s",
                partition.label,
                partition.chunk_start.date(),
                partition.chunk_end.date(),
                partition.path,
            )
            return ChunkResult(status="skipped", partition=partition, elapsed_seconds=elapsed)

        try:
            pipeline = build_candidate_pipeline(
                partition.chunk_start,
                partition.chunk_end,
                self.config.extraction.game,
                self.config.extraction.min_total_bet_amount,
                self.config.extraction.min_qualifying_players,
                timestamp_field=self.config.extraction.timestamp_field,
                match_start=partition.chunk_start if is_first_chunk else partition.chunk_start - BOUNDARY_PADDING,
                match_end=partition.chunk_end if is_last_chunk else partition.chunk_end + BOUNDARY_PADDING,
            )
            cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50)
            unknown_positions: Counter[str] = Counter()
            extraction_run_at = datetime.now(tz=timezone.utc)
            rows: list[dict[str, Any]] = []
            qualifying_players = 0
            groups_returned = 0

            for doc in cursor:
                groups_returned += 1
                qualifying_players += int(doc.get("qualifying_player_count") or 0)
                rows.append(
                    transform_grouped_doc(
                        doc,
                        self.config.extraction,
                        extraction_run_at=extraction_run_at,
                        unknown_positions=unknown_positions,
                    )
                )

            rows_written = write_candidate_rows(
                rows,
                partition.path,
                compression=self.config.output.parquet_compression,
                row_group_size=self.config.output.parquet_row_group_size,
                chunk_start=partition.chunk_start,
                chunk_end=partition.chunk_end,
            )
            unknown_count = sum(unknown_positions.values())
            elapsed = time.perf_counter() - started
            if unknown_positions:
                logger.warning(
                    "[chunk %s] unknown roulette positions skipped: %s",
                    partition.label,
                    dict(unknown_positions),
                )
            logger.info(
                "[chunk %s %s..%s] groups_returned=%d rows_written=%d qualifying_players=%d "
                "unknown_positions=%d elapsed=%.2fs path=%s",
                partition.label,
                partition.chunk_start.date(),
                partition.chunk_end.date(),
                groups_returned,
                rows_written,
                qualifying_players,
                unknown_count,
                elapsed,
                partition.path,
            )
            return ChunkResult(
                status="succeeded",
                partition=partition,
                groups_returned=groups_returned,
                rows_written=rows_written,
                qualifying_players=qualifying_players,
                unknown_positions=unknown_count,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.exception(
                "[chunk %s %s..%s] failed elapsed=%.2fs path=%s",
                partition.label,
                partition.chunk_start.date(),
                partition.chunk_end.date(),
                elapsed,
                partition.path,
            )
            return ChunkResult(
                status="failed",
                partition=partition,
                elapsed_seconds=elapsed,
                error=str(exc),
            )
