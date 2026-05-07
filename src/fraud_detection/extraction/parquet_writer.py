from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CANDIDATE_DRAW_SCHEMA = pa.schema(
    [
        pa.field("draw_id", pa.int64()),
        pa.field("trans_date_min", pa.timestamp("us", tz="UTC")),
        pa.field("trans_date_max", pa.timestamp("us", tz="UTC")),
        pa.field("qualifying_player_count", pa.int32()),
        pa.field("member_ids", pa.list_(pa.string())),
        pa.field("ccs_ids", pa.list_(pa.string())),
        pa.field("total_bet_amounts", pa.list_(pa.float64())),
        pa.field("win_points", pa.list_(pa.float64())),
        pa.field("coverage_bytes", pa.list_(pa.binary())),
        pa.field("amount_vector", pa.list_(pa.list_(pa.float64()))),
        pa.field("extraction_run_at", pa.timestamp("us", tz="UTC")),
    ]
)
CHUNK_START_METADATA_KEY = b"candidate_extraction.chunk_start"
CHUNK_END_METADATA_KEY = b"candidate_extraction.chunk_end"


@dataclass(frozen=True)
class PartitionInfo:
    chunk_start: datetime
    chunk_end: datetime
    path: Path

    @property
    def label(self) -> str:
        iso = self.chunk_start.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"


def resolve_partition_path(base_path: Path, chunk_start: datetime) -> Path:
    iso = chunk_start.isocalendar()
    return (
        base_path
        / f"year={iso.year:04d}"
        / f"month={chunk_start.month:02d}"
        / f"week={iso.week:02d}"
        / "draws.parquet"
    )


def write_candidate_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    compression: str = "zstd",
    row_group_size: int = 50_000,
    chunk_start: datetime | None = None,
    chunk_end: datetime | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = CANDIDATE_DRAW_SCHEMA
    if chunk_start is not None and chunk_end is not None:
        metadata = dict(schema.metadata or {})
        metadata[CHUNK_START_METADATA_KEY] = chunk_start.isoformat().encode("utf-8")
        metadata[CHUNK_END_METADATA_KEY] = chunk_end.isoformat().encode("utf-8")
        schema = schema.with_metadata(metadata)
    table = pa.Table.from_pylist(rows, schema=schema)
    temp_path = output_path.with_name(f"{output_path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(
        table,
        temp_path,
        compression=compression,
        row_group_size=row_group_size,
    )
    temp_path.replace(output_path)
    return table.num_rows


def read_partition_chunk_bounds(path: Path) -> tuple[str | None, str | None]:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    start = metadata.get(CHUNK_START_METADATA_KEY)
    end = metadata.get(CHUNK_END_METADATA_KEY)
    return (
        start.decode("utf-8") if start else None,
        end.decode("utf-8") if end else None,
    )
