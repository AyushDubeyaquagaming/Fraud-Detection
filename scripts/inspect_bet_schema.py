"""Inspect raw Mongo bet object schema for collusion feature feasibility."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.components.feature_engineering import parse_bets  # noqa: E402
from fraud_detection.constants.constants import (  # noqa: E402
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
)
from fraud_detection.logger import get_logger  # noqa: E402
from fraud_detection.utils.mongodb import get_mongo_collection  # noqa: E402

logger = get_logger(__name__)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "bet_schema_audit"


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def inspect_docs(docs: list[dict]) -> dict[str, Any]:
    field_counts: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = defaultdict(list)
    bet_count = 0
    docs_with_bets = 0

    for doc in docs:
        bets = parse_bets(doc.get("bets"))
        if bets:
            docs_with_bets += 1
        for bet in bets:
            if not isinstance(bet, dict):
                continue
            bet_count += 1
            for field, value in bet.items():
                field_counts[field] += 1
                type_counts[field][_type_name(value)] += 1
                if len(examples[field]) < 5:
                    examples[field].append(value)

    fields = {
        field: {
            "count": count,
            "types": dict(type_counts[field]),
            "examples": examples[field],
        }
        for field, count in sorted(field_counts.items())
    }
    field_set = set(field_counts)
    semantic_fields = field_set.difference({"_id"})
    has_type = any(field.lower() in {"type", "bet_type", "selection_type"} for field in field_set)
    has_number = "number" in semantic_fields
    parsed_by_current_template = ["number", "bet_amount"]
    ignored_fields = sorted(field_set.intersection({"_id"}))
    dropped_fields = sorted(semantic_fields.difference(parsed_by_current_template))

    if has_type:
        verdict = "bet.type present and parseable"
    elif has_number and semantic_fields.issubset({"number", "bet_amount"}):
        verdict = "only bet.number present"
    elif has_number:
        verdict = "number present with extra fields; inspect dropped_fields"
    else:
        verdict = "schema variant per draw type or unexpected bet shape"

    return {
        "sampled_docs": len(docs),
        "docs_with_bets": docs_with_bets,
        "bet_objects_seen": bet_count,
        "fields": fields,
        "parsed_by_current_template": parsed_by_current_template,
        "ignored_fields": ignored_fields,
        "dropped_fields": dropped_fields,
        "verdict": verdict,
    }


def sample_docs(sample_size: int) -> list[dict]:
    client, collection = get_mongo_collection(
        ENV_MONGODB_URI,
        ENV_MONGODB_DATABASE,
        ENV_MONGODB_COLLECTION,
    )
    try:
        return list(collection.aggregate([{"$sample": {"size": sample_size}}, {"$project": {"bets": 1}}]))
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    docs = sample_docs(args.sample_size)
    report = inspect_docs(docs)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Bet schema audit written to %s", report_path)
    print(f"verdict: {report['verdict']}")
    print(f"fields: {sorted(report['fields'])}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
