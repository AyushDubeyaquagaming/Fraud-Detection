from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fraud_detection.constants.constants import REPO_ROOT
from fraud_detection.extraction.ccs_profit_aggregator import build_ccs_daily_profit
from fraud_detection.utils.common import read_yaml


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build partitioned daily CCS-member profit parquet store.")
    parser.add_argument("--config", default="configs/ccs_profit.yaml")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild days even when they already exist in parquet.")
    parser.add_argument("--report-path", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = read_yaml(config_path)
    mongo_config = config.get("mongo", {}) or {}
    extraction_config = config.get("extraction", {}) or {}
    output_config = config.get("output", {}) or {}
    output_path = args.output_path or output_config.get("base_path", "data_store/ccs_daily_profit")
    summary = build_ccs_daily_profit(
        _parse_date(args.start_date),
        _parse_date(args.end_date),
        output_path,
        timestamp_field=str(extraction_config.get("timestamp_field", "trans_date")),
        compression=str(output_config.get("parquet_compression", "zstd")),
        uri_env_var=str(mongo_config.get("uri_env_var", "MONGODB_URI")),
        database_env_var=str(mongo_config.get("database_env_var", "MONGODB_DATABASE")),
        collection_env_var=str(mongo_config.get("collection_env_var", "MONGODB_COLLECTION_ROULETTE_REPORT")),
        force=bool(args.force),
        report_path=args.report_path,
    )
    print(json.dumps(summary.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
