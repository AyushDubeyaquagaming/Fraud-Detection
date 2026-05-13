from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from fraud_detection.constants.constants import REPO_ROOT


CCS_FEATURE_COLUMNS = [
    "ccs_profit_share_1d",
    "ccs_profit_share_7d",
    "ccs_total_profit_1d",
    "ccs_total_profit_7d",
    "ccs_member_count_1d",
    "ccs_member_count_7d",
    "ccs_high_concentration_1d",
    "ccs_high_concentration_7d",
    "ccs_solo_member_1d",
    "ccs_solo_member_7d",
]

_MISSING_ID_STRINGS = {"", "NAN", "NONE", "NULL", "NAT"}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _neutralize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CCS_FEATURE_COLUMNS:
        out[column] = 0.0
    return _order_feature_columns(out)


def _order_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base_columns = [column for column in frame.columns if column not in CCS_FEATURE_COLUMNS]
    for column in CCS_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[base_columns + CCS_FEATURE_COLUMNS]


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    invalid = normalized.isna() | normalized.isin(_MISSING_ID_STRINGS)
    return normalized.mask(invalid, other=None).astype(object)


def _normalize_member_rows(member_draw_df: pd.DataFrame) -> pd.DataFrame:
    out = member_draw_df.copy()
    out["member_id"] = _normalize_identifier_series(out["member_id"])
    out["ccs_id"] = _normalize_identifier_series(out["ccs_id"])
    out["_draw_day"] = pd.to_datetime(out["draw_date"], errors="coerce", utc=True).dt.normalize()
    return out


def _prepare_profit_frame(profit: pd.DataFrame) -> pd.DataFrame:
    if profit.empty:
        return pd.DataFrame(columns=["ccs_id", "member_id", "profit_date", "daily_profit"])
    out = profit.copy()
    out["member_id"] = _normalize_identifier_series(out["member_id"])
    out["ccs_id"] = _normalize_identifier_series(out["ccs_id"])
    out["profit_date"] = pd.to_datetime(out["profit_date"], errors="coerce", utc=True).dt.normalize()
    out["daily_profit"] = pd.to_numeric(out.get("daily_profit", 0.0), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["ccs_id", "member_id", "profit_date"])
    return out[["ccs_id", "member_id", "profit_date", "daily_profit"]].copy()


def prepare_profit_frame(profit: pd.DataFrame) -> pd.DataFrame:
    return _prepare_profit_frame(profit)


def _read_relevant_profit_rows(
    profit_path: Path,
    *,
    ccs_ids: set[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not profit_path.exists() or not ccs_ids:
        return pd.DataFrame(columns=["ccs_id", "member_id", "profit_date", "daily_profit"])
    dataset = ds.dataset(profit_path, format="parquet", partitioning="hive")
    filters = (ds.field("profit_date") >= start_date.date()) & (ds.field("profit_date") <= end_date.date())
    chunks: list[pd.DataFrame] = []
    for batch in dataset.scanner(filter=filters, batch_size=100_000).to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if "ccs_id" not in frame.columns:
            continue
        frame["ccs_id"] = frame["ccs_id"].astype(str).str.strip().str.upper()
        frame = frame.loc[frame["ccs_id"].isin(ccs_ids)].copy()
        if not frame.empty:
            chunks.append(frame)
    if not chunks:
        return pd.DataFrame(columns=["ccs_id", "member_id", "profit_date", "daily_profit"])
    return _prepare_profit_frame(pd.concat(chunks, ignore_index=True))


def _attach_from_profit_frame(
    member_draw_df: pd.DataFrame,
    *,
    profit_frame: pd.DataFrame,
    windows_days: list[int],
    concentration_threshold: float,
) -> pd.DataFrame:
    out = _normalize_member_rows(member_draw_df)
    valid = out.loc[out["ccs_id"].notna() & out["member_id"].notna() & out["_draw_day"].notna()].copy()
    if valid.empty:
        return _neutralize(out.drop(columns=["_draw_day"], errors="ignore"))

    windows = sorted({int(window) for window in windows_days if int(window) > 0}) or [1, 7]
    profit = _prepare_profit_frame(profit_frame)
    if profit.empty:
        return _neutralize(out.drop(columns=["_draw_day"], errors="ignore"))
    max_window = max(windows)
    min_profit_date = valid["_draw_day"].min() - pd.Timedelta(days=max_window - 1)
    max_profit_date = valid["_draw_day"].max()
    profit = profit.loc[
        profit["ccs_id"].isin(set(valid["ccs_id"]))
        & profit["profit_date"].between(min_profit_date, max_profit_date, inclusive="both")
    ].copy()
    if profit.empty:
        return _neutralize(out.drop(columns=["_draw_day"], errors="ignore"))

    row_keys = valid[["ccs_id", "member_id", "_draw_day"]].drop_duplicates().reset_index(drop=True)
    for window in windows:
        share_col = f"ccs_profit_share_{window}d"
        total_col = f"ccs_total_profit_{window}d"
        count_col = f"ccs_member_count_{window}d"
        high_col = f"ccs_high_concentration_{window}d"
        solo_col = f"ccs_solo_member_{window}d"
        out[share_col] = 0.0
        out[total_col] = 0.0
        out[count_col] = 0.0
        out[high_col] = 0.0
        out[solo_col] = 0.0

        ccs_keys = row_keys[["ccs_id", "_draw_day"]].drop_duplicates().reset_index(drop=True)
        ccs_keys["_ccs_key"] = ccs_keys.index
        ccs_window = _expand_window_keys(ccs_keys, window, key_column="_ccs_key")
        ccs_profit = ccs_window.merge(
            profit[["ccs_id", "member_id", "profit_date", "daily_profit"]],
            on=["ccs_id", "profit_date"],
            how="left",
        )
        ccs_summary = (
            ccs_profit.groupby("_ccs_key", as_index=False)
            .agg(
                **{
                    total_col: ("daily_profit", "sum"),
                    count_col: ("member_id", "nunique"),
                }
            )
        )

        member_keys = row_keys.copy()
        member_keys["_member_key"] = member_keys.index
        member_window = _expand_window_keys(member_keys, window, key_column="_member_key")
        member_profit = member_window.merge(
            profit[["ccs_id", "member_id", "profit_date", "daily_profit"]],
            on=["ccs_id", "member_id", "profit_date"],
            how="left",
        )
        member_summary = (
            member_profit.groupby("_member_key", as_index=False)["daily_profit"]
            .sum()
            .rename(columns={"daily_profit": "_member_profit"})
        )

        features = (
            member_keys.merge(ccs_keys, on=["ccs_id", "_draw_day"], how="left")
            .merge(ccs_summary, on="_ccs_key", how="left")
            .merge(member_summary, on="_member_key", how="left")
        )
        features[total_col] = pd.to_numeric(features[total_col], errors="coerce").fillna(0.0)
        features[count_col] = pd.to_numeric(features[count_col], errors="coerce").fillna(0.0)
        features["_member_profit"] = pd.to_numeric(features["_member_profit"], errors="coerce").fillna(0.0)
        features[share_col] = 0.0
        positive_total = features[total_col] > 0
        features.loc[positive_total, share_col] = (
            features.loc[positive_total, "_member_profit"].clip(lower=0.0) / features.loc[positive_total, total_col]
        ).clip(upper=1.0)
        features[high_col] = (
            positive_total & features[count_col].ge(2) & features[share_col].ge(float(concentration_threshold))
        ).astype(float)
        features[solo_col] = features[count_col].eq(1).astype(float)

        feature_index = pd.MultiIndex.from_frame(features[["ccs_id", "member_id", "_draw_day"]])
        aligned = features.set_index(feature_index)[[share_col, total_col, count_col, high_col, solo_col]]
        valid_index = pd.MultiIndex.from_frame(valid[["ccs_id", "member_id", "_draw_day"]])
        out.loc[valid.index, [share_col, total_col, count_col, high_col, solo_col]] = (
            aligned.reindex(valid_index).fillna(0.0).to_numpy()
        )

    for column in CCS_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return _order_feature_columns(out.drop(columns=["_draw_day"], errors="ignore"))


def _expand_window_keys(keys: pd.DataFrame, window: int, *, key_column: str) -> pd.DataFrame:
    frames = []
    for offset in range(window):
        frame = keys.copy()
        frame["profit_date"] = frame["_draw_day"] - pd.Timedelta(days=offset)
        frames.append(frame[[key_column, "ccs_id", "member_id", "profit_date"]] if "member_id" in frame.columns else frame[[key_column, "ccs_id", "profit_date"]])
    return pd.concat(frames, ignore_index=True) if frames else keys.iloc[0:0].copy()


def attach_ccs_concentration_features_from_frame(
    member_draw_df: pd.DataFrame,
    *,
    profit_frame: pd.DataFrame,
    windows_days: list[int],
    concentration_threshold: float,
) -> pd.DataFrame:
    if member_draw_df.empty:
        return _neutralize(member_draw_df)
    if "ccs_id" not in member_draw_df.columns or "draw_date" not in member_draw_df.columns:
        return _neutralize(member_draw_df)
    return _attach_from_profit_frame(
        member_draw_df,
        profit_frame=profit_frame,
        windows_days=windows_days,
        concentration_threshold=concentration_threshold,
    )


def load_relevant_profit_rows(
    member_draw_df: pd.DataFrame,
    *,
    ccs_profit_path: str | Path,
    windows_days: list[int],
) -> pd.DataFrame:
    if member_draw_df.empty or "ccs_id" not in member_draw_df.columns or "draw_date" not in member_draw_df.columns:
        return pd.DataFrame(columns=["ccs_id", "member_id", "profit_date", "daily_profit"])
    rows = _normalize_member_rows(member_draw_df)
    rows = rows.loc[rows["ccs_id"].notna() & rows["_draw_day"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(columns=["ccs_id", "member_id", "profit_date", "daily_profit"])
    windows = sorted({int(window) for window in windows_days if int(window) > 0}) or [1, 7]
    max_window = max(windows)
    return _read_relevant_profit_rows(
        _resolve_path(ccs_profit_path),
        ccs_ids=set(rows["ccs_id"].astype(str)),
        start_date=rows["_draw_day"].min() - pd.Timedelta(days=max_window - 1),
        end_date=rows["_draw_day"].max(),
    )


def attach_ccs_concentration_features(
    member_draw_df: pd.DataFrame,
    *,
    ccs_profit_path: str | Path,
    windows_days: list[int],
    concentration_threshold: float,
) -> pd.DataFrame:
    """
    Attach CCS concentration features to member-draw rows.

    Window semantics are calendar-day inclusive: a 1-day window uses the draw's
    profit_date only; a 7-day window uses draw_date - 6 days through draw_date.
    """
    if member_draw_df.empty:
        return _neutralize(member_draw_df)
    if "ccs_id" not in member_draw_df.columns or "draw_date" not in member_draw_df.columns:
        return _neutralize(member_draw_df)
    profit = load_relevant_profit_rows(
        member_draw_df,
        ccs_profit_path=ccs_profit_path,
        windows_days=windows_days,
    )
    return attach_ccs_concentration_features_from_frame(
        member_draw_df,
        profit_frame=profit,
        windows_days=windows_days,
        concentration_threshold=concentration_threshold,
    )
