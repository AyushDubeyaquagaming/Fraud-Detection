from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

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
from fraud_detection.utils.mongodb import get_serving_mongo_collection

logger = get_logger(__name__)


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

    @staticmethod
    def _to_arrow_filter_bound(value: datetime, arrow_type: pa.DataType) -> object:
        timestamp = pd.Timestamp(value)
        if pa.types.is_timestamp(arrow_type):
            if arrow_type.tz:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                else:
                    timestamp = timestamp.tz_convert(arrow_type.tz)
            else:
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            return timestamp.to_pydatetime()
        if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            return timestamp.date()
        return timestamp.to_pydatetime()

    def _build_coarse_parquet_filter(
        self,
        dataset: ds.Dataset,
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool,
    ):
        available_ts = self._available_timestamp_columns(dataset.schema.names)
        if not available_ts:
            return None
        expression = None
        for coarse_field in available_ts:
            arrow_type = dataset.schema.field(coarse_field).type
            lower = ds.field(coarse_field) >= self._to_arrow_filter_bound(start, arrow_type)
            upper_bound = self._to_arrow_filter_bound(end, arrow_type)
            upper = ds.field(coarse_field) <= upper_bound if inclusive_end else ds.field(coarse_field) < upper_bound
            clause = lower & upper
            expression = clause if expression is None else (expression | clause)
        return expression

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

        schema = ds.dataset(self.training_parquet_path, format="parquet").schema
        timestamp_columns = self._available_timestamp_columns(schema.names)
        if not timestamp_columns:
            return None, None
        df = pd.read_parquet(self.training_parquet_path, columns=timestamp_columns)
        valid = _normalize_timestamp(df).dropna()
        if valid.empty:
            self._parquet_start_date = None
            self._parquet_end_date = None
        else:
            self._parquet_start_date = valid.min().to_pydatetime()
            self._parquet_end_date = valid.max().to_pydatetime()
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

    def _pull_from_parquet(
        self,
        member_id: str,
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool,
    ) -> pd.DataFrame:
        dataset = ds.dataset(self.training_parquet_path, format="parquet")
        parquet_columns = [col for col in FEATURE_ENGINEERING_RAW_COLUMNS if col in dataset.schema.names]
        for col in self._available_timestamp_columns(dataset.schema.names):
            if col not in parquet_columns:
                parquet_columns.append(col)
        coarse_filter = self._build_coarse_parquet_filter(dataset, start, end, inclusive_end=inclusive_end)
        table = dataset.to_table(columns=parquet_columns, filter=coarse_filter)
        df = table.to_pandas()
        return self._post_filter_rows(df, member_id, start, end, inclusive_end=inclusive_end)

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
                for candidate in dict.fromkeys((self.timestamp_field, *self.timestamp_candidates))
            ]
            query = {
                "member_id": {"$regex": member_regex, "$options": "i"},
                "$or": ts_clauses,
            }
            docs = list(collection.find(query))
            if not docs:
                return pd.DataFrame()

            df = pd.DataFrame(docs)
            if "bets" in df.columns:
                import json

                df["bets"] = df["bets"].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
            return self._post_filter_rows(df, member_id, start, end, inclusive_end=False)
        except Exception as exc:
            raise MongoWindowFetchError(f"MongoDB window fetch failed: {exc}") from exc
