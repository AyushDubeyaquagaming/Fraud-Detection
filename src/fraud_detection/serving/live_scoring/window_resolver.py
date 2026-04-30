from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq

from fraud_detection.components.feature_engineering import (
    FEATURE_ENGINEERING_RAW_COLUMNS,
    TIMESTAMP_CANDIDATES,
    _normalize_timestamp,
)
from fraud_detection.constants.constants import (
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
)
from fraud_detection.logger import get_logger
from fraud_detection.utils.mongodb import MONGO_PROJECTION, _serialize_bets, get_serving_mongo_collection

from .parquet_metadata import read_training_parquet_bounds

logger = get_logger(__name__)

# Per-request streaming and safety caps. Bounded so a single live request can
# never accidentally materialize an entire parquet window or Mongo cursor.
LIVE_MONGO_BATCH_SIZE = 10_000
LIVE_MONGO_MAX_DOCS = 100_000

# Cap retained matched rows per request as a defense-in-depth limit. A single
# member's draws over a few weeks should never approach this — if we hit it,
# something is wrong upstream and we'd rather log+truncate than OOM.
LIVE_PARQUET_MAX_RETAINED_ROWS = 500_000

# Mongo timestamp candidates — only fields that actually exist in the source
# collection (per MONGO_PROJECTION). The longer TIMESTAMP_CANDIDATES list is
# parquet-only (it includes post-FE artifacts like `ts` and `*.$date`).
MONGO_TIMESTAMP_CANDIDATES = ("trans_date", "createdAt", "updatedAt")


class MongoWindowFetchError(RuntimeError):
    """Raised when a required Mongo fetch fails for a live/historical request."""


@dataclass
class WindowData:
    raw_df: pd.DataFrame
    window_start: datetime
    window_end: datetime
    parquet_rows: int
    mongo_rows: int
    data_sources: list[str]
    training_parquet_start_date: datetime | None
    training_parquet_end_date: datetime | None


class WindowResolver:
    """Resolve one member's request window against parquet and Mongo sources."""

    def __init__(
        self,
        training_parquet_path: Path,
        timestamp_field: str = "trans_date",
        timestamp_candidates: tuple[str, ...] | None = None,
        parquet_start_date: datetime | None = None,
        parquet_end_date: datetime | None = None,
    ):
        self.training_parquet_path = Path(training_parquet_path)
        self.timestamp_field = timestamp_field
        self.timestamp_candidates = tuple(timestamp_candidates or TIMESTAMP_CANDIDATES)
        self._parquet_start_date = parquet_start_date
        self._parquet_end_date = parquet_end_date

    @staticmethod
    def _normalize_member_id(value: object) -> str:
        return str(value).strip().upper()

    def _available_timestamp_columns(self, columns: list[str]) -> list[str]:
        ordered = []
        for col in (self.timestamp_field, *self.timestamp_candidates):
            if col in columns and col not in ordered:
                ordered.append(col)
        return ordered

    def _post_filter_rows(
        self,
        df: pd.DataFrame,
        member_id: str,
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        working = df.copy()
        working["_member_id_norm"] = working["member_id"].astype(str).str.strip().str.upper()
        working["_ts"] = _normalize_timestamp(working)
        end_mask = working["_ts"] <= end if inclusive_end else working["_ts"] < end
        filtered = working.loc[
            working["_member_id_norm"].eq(self._normalize_member_id(member_id))
            & working["_ts"].notna()
            & (working["_ts"] >= start)
            & end_mask
        ].copy()
        return filtered.drop(columns=["_member_id_norm", "_ts"], errors="ignore")

    def _determine_parquet_date_range(self) -> tuple[datetime | None, datetime | None]:
        if self._parquet_start_date is not None and self._parquet_end_date is not None:
            return self._parquet_start_date, self._parquet_end_date

        # Read bounds from the parquet footer (row-group statistics) — no row
        # data is loaded. Falls back to a streamed single-column scan only if
        # statistics are missing. Never holds the full parquet in memory.
        start_date, end_date = read_training_parquet_bounds(self.training_parquet_path)
        self._parquet_start_date = start_date
        self._parquet_end_date = end_date
        return self._parquet_start_date, self._parquet_end_date

    def resolve(self, member_id: str, start_date: datetime, end_date: datetime) -> WindowData:
        parquet_start, parquet_end = self._determine_parquet_date_range()
        data_sources: list[str] = []
        parquet_df = pd.DataFrame()
        mongo_frames: list[pd.DataFrame] = []

        if parquet_start is not None and parquet_end is not None:
            parquet_window_start = max(start_date, parquet_start)
            parquet_window_end = min(end_date, parquet_end)
            include_parquet_boundary = end_date > parquet_end and parquet_window_start <= parquet_window_end
            if parquet_window_start < parquet_window_end or include_parquet_boundary:
                parquet_df = self._pull_from_parquet(
                    member_id,
                    parquet_window_start,
                    parquet_window_end,
                    inclusive_end=include_parquet_boundary,
                )
                if not parquet_df.empty:
                    data_sources.append("training_parquet")

            if start_date < parquet_start:
                pre_end = min(end_date, parquet_start)
                pre_df = self._pull_from_mongo(member_id, start_date, pre_end, start_inclusive=True)
                if not pre_df.empty:
                    mongo_frames.append(pre_df)
                    data_sources.append("mongo_historical")

            if end_date > parquet_end:
                post_start = max(start_date, parquet_end)
                post_df = self._pull_from_mongo(member_id, post_start, end_date, start_inclusive=False)
                if not post_df.empty:
                    mongo_frames.append(post_df)
                    data_sources.append("mongo_delta")
        else:
            mongo_df = self._pull_from_mongo(member_id, start_date, end_date, start_inclusive=True)
            if not mongo_df.empty:
                mongo_frames.append(mongo_df)
                data_sources.append("mongo_historical")

        mongo_df = pd.concat(mongo_frames, ignore_index=True) if mongo_frames else pd.DataFrame()
        combined = pd.concat([parquet_df, mongo_df], ignore_index=True)
        if not combined.empty:
            combined["_member_id_norm"] = combined["member_id"].astype(str).str.strip().str.upper()
            combined["_resolved_ts"] = _normalize_timestamp(combined)
            combined = combined.drop_duplicates(subset=["_member_id_norm", "draw_id"], keep="first")
            combined = combined.sort_values(["_resolved_ts", "draw_id"], na_position="last").reset_index(drop=True)
            combined = combined.drop(columns=["_member_id_norm", "_resolved_ts"], errors="ignore")

        return WindowData(
            raw_df=combined,
            window_start=start_date,
            window_end=end_date,
            parquet_rows=len(parquet_df),
            mongo_rows=len(mongo_df),
            data_sources=data_sources,
            training_parquet_start_date=parquet_start,
            training_parquet_end_date=parquet_end,
        )

    @staticmethod
    def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _row_group_overlaps_window(
        self,
        rg_metadata,
        ts_column_indices: list[int],
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool,
    ) -> bool:
        """Decide if a row group could contain rows in [start, end).

        Uses parquet column statistics on every available timestamp candidate.
        If ANY candidate's [min,max] overlaps the request window, the row group
        is in-scope. If statistics are missing for every candidate, we must
        include the row group (cannot prove it's empty).
        """
        start_ts = self._to_utc_timestamp(start)
        end_ts = self._to_utc_timestamp(end)
        any_stats = False
        for ts_idx in ts_column_indices:
            stats = rg_metadata.column(ts_idx).statistics
            if stats is None or not stats.has_min_max:
                continue
            any_stats = True
            rg_min = pd.to_datetime(stats.min, utc=True, errors="coerce")
            rg_max = pd.to_datetime(stats.max, utc=True, errors="coerce")
            if pd.isna(rg_min) or pd.isna(rg_max):
                continue
            if rg_max < start_ts:
                continue
            if inclusive_end:
                if rg_min > end_ts:
                    continue
            else:
                if rg_min >= end_ts:
                    continue
            return True
        return not any_stats

    def _pull_from_parquet(
        self,
        member_id: str,
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool,
    ) -> pd.DataFrame:
        """Read this member's draws from the promoted training parquet without
        ever holding more than one row group in memory at a time.

        Why a manual row-group loop: pyarrow's `dataset.scanner` with a combined
        ts+member_id filter still has to materialize each in-window row group
        in an internal buffer to evaluate the row-level member_id predicate
        (parquet column statistics on `member_id` aren't selective because the
        data is sorted by time, not member). On a 15 GB / 4045-row-group
        training parquet the resulting buffering caused +1.2 GB RSS per request.
        Iterating row groups by hand gives explicit control of the load/filter/
        free cycle so peak RAM stays at one row-group's worth (~30 MB).
        """
        pf = pq.ParquetFile(str(self.training_parquet_path))
        schema_names = pf.schema_arrow.names

        parquet_columns = [col for col in FEATURE_ENGINEERING_RAW_COLUMNS if col in schema_names]
        for col in self._available_timestamp_columns(schema_names):
            if col not in parquet_columns:
                parquet_columns.append(col)
        if "member_id" in schema_names and "member_id" not in parquet_columns:
            parquet_columns.append("member_id")

        ts_column_indices = [
            pf.schema_arrow.get_field_index(col)
            for col in self._available_timestamp_columns(schema_names)
        ]
        ts_column_indices = [i for i in ts_column_indices if i >= 0]

        normalized = self._normalize_member_id(member_id)
        frames: list[pd.DataFrame] = []
        retained_rows = 0
        truncated = False

        for rg_idx in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(rg_idx)
            if not self._row_group_overlaps_window(
                rg_meta, ts_column_indices, start, end, inclusive_end=inclusive_end
            ):
                continue

            table = pf.read_row_group(rg_idx, columns=parquet_columns)
            df = table.to_pandas()
            del table

            if "member_id" not in df.columns or df.empty:
                continue
            normalized_ids = df["member_id"].astype(str).str.strip().str.upper()
            df = df.loc[normalized_ids == normalized]
            if df.empty:
                continue

            frames.append(df)
            retained_rows += len(df)
            if retained_rows >= LIVE_PARQUET_MAX_RETAINED_ROWS:
                logger.warning(
                    "Live parquet pull hit retention cap of %d rows for member=%s "
                    "window=[%s, %s); truncating.",
                    LIVE_PARQUET_MAX_RETAINED_ROWS,
                    member_id,
                    start.isoformat(),
                    end.isoformat(),
                )
                truncated = True
                break

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        if truncated:
            result.attrs["truncated"] = True
        return self._post_filter_rows(result, member_id, start, end, inclusive_end=inclusive_end)

    def _mongo_timestamp_candidates(self) -> list[str]:
        """Return Mongo timestamp candidates, scoped to fields that actually exist
        in the source collection.

        The longer parquet-side `TIMESTAMP_CANDIDATES` list includes post-FE
        artifacts like `ts` and `*.$date` which are never stored in Mongo. Each
        added candidate forces an additional clause in the `$or` query — and
        without an index on every clause that becomes a collection scan. Restrict
        to the three fields in `MONGO_PROJECTION` (trans_date, createdAt,
        updatedAt), with the configured `timestamp_field` first.
        """
        ordered: list[str] = []
        for candidate in (self.timestamp_field, *MONGO_TIMESTAMP_CANDIDATES):
            if candidate in MONGO_TIMESTAMP_CANDIDATES and candidate not in ordered:
                ordered.append(candidate)
        if not ordered and self.timestamp_field:
            ordered.append(self.timestamp_field)
        return ordered

    def _pull_from_mongo(
        self,
        member_id: str,
        start: datetime,
        end: datetime,
        *,
        start_inclusive: bool,
    ) -> pd.DataFrame:
        if start > end or (start == end and start_inclusive):
            return pd.DataFrame()

        try:
            collection = get_serving_mongo_collection(
                ENV_MONGODB_URI,
                ENV_MONGODB_DATABASE,
                ENV_MONGODB_COLLECTION,
            )
        except Exception as exc:
            raise MongoWindowFetchError(f"MongoDB connection failed: {exc}") from exc

        try:
            start_operator = "$gte" if start_inclusive else "$gt"
            member_regex = rf"^\s*{re.escape(self._normalize_member_id(member_id))}\s*$"
            ts_clauses = [
                {candidate: {start_operator: start, "$lt": end}}
                for candidate in self._mongo_timestamp_candidates()
            ]
            query = {
                "member_id": {"$regex": member_regex, "$options": "i"},
            }
            if len(ts_clauses) == 1:
                query.update(ts_clauses[0])
            elif ts_clauses:
                query["$or"] = ts_clauses

            cursor = collection.find(query, MONGO_PROJECTION).batch_size(LIVE_MONGO_BATCH_SIZE)
            docs: list[dict] = []
            try:
                for doc in cursor:
                    docs.append(doc)
                    if len(docs) >= LIVE_MONGO_MAX_DOCS:
                        logger.warning(
                            "Live Mongo fetch hit safety cap of %d docs for member=%s "
                            "window=[%s, %s); truncating.",
                            LIVE_MONGO_MAX_DOCS,
                            member_id,
                            start.isoformat(),
                            end.isoformat(),
                        )
                        break
            finally:
                cursor.close()

            if not docs:
                return pd.DataFrame()

            df = pd.DataFrame(docs)
            df = df.drop(columns=["_id"], errors="ignore")
            if "bets" in df.columns:
                df["bets"] = df["bets"].apply(_serialize_bets)
            return self._post_filter_rows(df, member_id, start, end, inclusive_end=False)
        except MongoWindowFetchError:
            raise
        except Exception as exc:
            raise MongoWindowFetchError(f"MongoDB window fetch failed: {exc}") from exc
