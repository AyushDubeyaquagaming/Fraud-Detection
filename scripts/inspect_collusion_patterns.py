"""Explore labelled same-draw roulette collusion patterns.

This is the script form of the Phase B notebook. It pulls known fraud-member
transactions, finds labelled draws with multiple fraud members, compares their
board coverage/overlap with random same-draw cohorts, and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.analysis.collusion import (  # noqa: E402
    compute_draw_cohort_metrics,
    compute_draw_metrics_table,
    decision_verdict,
)
from fraud_detection.constants.constants import (  # noqa: E402
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
)
from fraud_detection.logger import get_logger  # noqa: E402
from fraud_detection.utils.mongodb import (  # noqa: E402
    MONGO_PROJECTION,
    build_query_batches_from_strategy,
    get_mongo_collection,
    pull_query_batches_to_dataframe,
)

logger = get_logger(__name__)

DEFAULT_FRAUD_CSV = REPO_ROOT / "ROULET CHEATING DATA.csv"
DEFAULT_CACHE_PATH = REPO_ROOT / "data_cache" / "collusion_exploration_pull.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "collusion_exploration"

def load_fraud_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels.columns = [c.strip().lower() for c in labels.columns]
    labels["member_id"] = labels["member_id"].astype(str).str.strip().str.upper()
    labels["draw_id"] = pd.to_numeric(labels["draw_id"], errors="coerce").astype("Int64")
    labels["fraud_date"] = pd.to_datetime(labels.get("date"), errors="coerce", utc=True)
    return labels.dropna(subset=["draw_id", "member_id"]).copy()


def pull_fraud_member_transactions(labels: pd.DataFrame, cache_path: Path, refresh_cache: bool) -> pd.DataFrame:
    if cache_path.exists() and not refresh_cache:
        logger.info("Loading cached fraud-member transactions from %s", cache_path)
        return pd.read_parquet(cache_path)

    member_ids = sorted(labels["member_id"].dropna().astype(str).str.upper().unique())
    query_filters = build_query_batches_from_strategy(
        "member_list",
        {"member_ids_source": "inline", "member_ids": member_ids},
    )
    df = pull_query_batches_to_dataframe(
        uri_env_var=ENV_MONGODB_URI,
        db_env_var=ENV_MONGODB_DATABASE,
        collection_env_var=ENV_MONGODB_COLLECTION,
        query_filters=query_filters,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def pull_draw_transactions(draw_ids: list[int]) -> pd.DataFrame:
    if not draw_ids:
        return pd.DataFrame()
    client, collection = get_mongo_collection(ENV_MONGODB_URI, ENV_MONGODB_DATABASE, ENV_MONGODB_COLLECTION)
    try:
        cursor = collection.find({"draw_id": {"$in": draw_ids}}, MONGO_PROJECTION)
        docs = list(cursor)
    finally:
        client.close()
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")


def pull_draw_transactions_from_parquet(raw_parquet: Path, draw_ids: list[int]) -> pd.DataFrame:
    if not draw_ids:
        return pd.DataFrame()
    draw_id_set = set(int(draw_id) for draw_id in draw_ids)
    parquet_file = pq.ParquetFile(raw_parquet)
    columns = [col for col in MONGO_PROJECTION if col in parquet_file.schema_arrow.names]
    frames = []
    for batch in parquet_file.iter_batches(columns=columns, batch_size=250_000):
        df = batch.to_pandas()
        if "draw_id" not in df.columns:
            continue
        draw_values = pd.to_numeric(df["draw_id"], errors="coerce").astype("Int64")
        mask = draw_values.isin(draw_id_set)
        if mask.any():
            frames.append(df.loc[mask].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def labelled_fraud_draw_rows(labels: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    working = transactions.copy()
    working["member_id"] = working["member_id"].astype(str).str.strip().str.upper()
    working["draw_id"] = pd.to_numeric(working["draw_id"], errors="coerce").astype("Int64")
    keys = set(zip(labels["draw_id"].astype(int), labels["member_id"]))
    mask = [(int(draw_id), member_id) in keys for draw_id, member_id in zip(working["draw_id"], working["member_id"])]
    return working.loc[mask].copy()


def random_same_draw_baseline(draw_rows: pd.DataFrame, fraud_metrics: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    if draw_rows.empty or fraud_metrics.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    rows = []
    for _, metric in fraud_metrics.iterrows():
        draw_id = int(metric["draw_id"])
        cohort_size = int(metric["cohort_size"])
        candidates = draw_rows.loc[pd.to_numeric(draw_rows["draw_id"], errors="coerce").eq(draw_id)]
        unique_members = candidates["member_id"].astype(str).str.strip().str.upper().drop_duplicates()
        if len(unique_members) < cohort_size or cohort_size < 2:
            continue
        chosen = set(rng.choice(unique_members.to_numpy(), size=cohort_size, replace=False))
        sample = candidates.loc[candidates["member_id"].astype(str).str.strip().str.upper().isin(chosen)]
        metrics = compute_draw_cohort_metrics(sample)
        metrics["draw_id"] = draw_id
        rows.append(metrics)
    return pd.DataFrame(rows)


def dataframe_summary(df: pd.DataFrame, columns: list[str]) -> dict:
    if df.empty:
        return {}
    summary = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[col] = {
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraud-csv", type=Path, default=DEFAULT_FRAUD_CSV)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--raw-parquet",
        type=Path,
        default=None,
        help="Optional raw training parquet for same-draw baseline rows. Avoids slow draw_id Mongo queries.",
    )
    args = parser.parse_args()

    labels = load_fraud_labels(args.fraud_csv)
    fraud_txns = pull_fraud_member_transactions(labels, args.cache_path, args.refresh_cache)
    fraud_draw_rows = labelled_fraud_draw_rows(labels, fraud_txns)
    fraud_metrics_all = compute_draw_metrics_table(fraud_draw_rows)
    fraud_metrics = fraud_metrics_all.loc[fraud_metrics_all["cohort_size"] >= 2].copy()

    draw_ids = sorted(fraud_metrics["draw_id"].dropna().astype(int).unique().tolist())
    if args.raw_parquet is not None:
        all_draw_rows = pull_draw_transactions_from_parquet(args.raw_parquet, draw_ids)
    else:
        all_draw_rows = pull_draw_transactions(draw_ids)
    if not all_draw_rows.empty:
        all_draw_rows["member_id"] = all_draw_rows["member_id"].astype(str).str.strip().str.upper()
    baseline_metrics = random_same_draw_baseline(all_draw_rows, fraud_metrics, args.random_seed)
    decision = decision_verdict(fraud_metrics_all, baseline_metrics)

    metric_cols = [
        "cohort_size",
        "union_position_coverage",
        "mean_pairwise_jaccard",
        "total_cohort_stake",
        "mean_member_stake",
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fraud_csv": str(args.fraud_csv),
        "fraud_label_rows": int(len(labels)),
        "fraud_label_members": int(labels["member_id"].nunique()),
        "fraud_member_transaction_rows": int(len(fraud_txns)),
        "labelled_fraud_draw_rows": int(len(fraud_draw_rows)),
        "fraud_draws_total": int(len(fraud_metrics_all)),
        "fraud_draws_with_2plus_members": int(len(fraud_metrics)),
        "baseline_draws": int(len(baseline_metrics)),
        "decision": decision,
        "fraud_metric_summary": dataframe_summary(fraud_metrics, metric_cols),
        "baseline_metric_summary": dataframe_summary(baseline_metrics, metric_cols),
        "top_fraud_draws": fraud_metrics.sort_values(
            ["union_position_coverage", "cohort_size"],
            ascending=False,
        ).head(25).to_dict(orient="records"),
    }

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    fraud_metrics.to_csv(out_dir / "fraud_draw_metrics.csv", index=False)
    baseline_metrics.to_csv(out_dir / "baseline_draw_metrics.csv", index=False)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"verdict: {decision['verdict']}")
    print(f"reason: {decision['reason']}")
    print(f"fraud_draws_with_2plus_members: {len(fraud_metrics)}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
