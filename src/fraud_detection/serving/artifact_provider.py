from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.constants.constants import MODEL_BUNDLE_FILE, PROMOTION_METADATA_FILE, REPO_ROOT, RUN_METADATA_FILE
from fraud_detection.utils.common import load_joblib, read_json


@dataclass(frozen=True)
class ArtifactBundle:
    stage2_holdout_predictions_df: pd.DataFrame
    serving_manifest: dict[str, Any]
    snapshot_metadata: dict[str, Any]
    promotion_metadata: dict[str, Any]
    evaluation_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    snapshot_available: bool
    snapshot_reason: str | None
    loaded_at: datetime
    source_run_id: str
    promoted_at: str | None
    evaluated_at: str | None
    model_version: str
    model_bundle: dict[str, Any] | None = None
    partnership_table_df: pd.DataFrame | None = None
    ccs_concentration_table_df: pd.DataFrame | None = None
    training_raw_parquet_path: Path | None = None
    training_parquet_start_date: datetime | None = None
    training_parquet_end_date: datetime | None = None


class ArtifactProvider(ABC):
    @abstractmethod
    def load(self) -> ArtifactBundle:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class LocalDiskArtifactProvider(ArtifactProvider):
    def __init__(self, current_dir: Path, manifest_file: str = "serving_manifest.json", default_model_version: str = "partnership_v1", repo_root: Path = REPO_ROOT):
        self.current_dir = Path(current_dir)
        self.repo_root = Path(repo_root)
        self.default_model_version = default_model_version
        self.manifest_path = self.current_dir / manifest_file

    def is_available(self) -> bool:
        return self.manifest_path.exists()

    def load(self) -> ArtifactBundle:
        if not self.is_available():
            raise FileNotFoundError(f"Serving manifest not found at {self.manifest_path}")
        manifest = read_json(self.manifest_path)
        run_dir = Path(manifest["run_dir"])
        promotion_metadata = read_json(self.current_dir / PROMOTION_METADATA_FILE) if (self.current_dir / PROMOTION_METADATA_FILE).exists() else {}
        run_metadata = read_json(run_dir / RUN_METADATA_FILE) if (run_dir / RUN_METADATA_FILE).exists() else {}
        evaluation_path = self.current_dir / "evaluation_report.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else {}
        bundle_path = self.current_dir / manifest.get("model_bundle_file", MODEL_BUNDLE_FILE)
        model_bundle = load_joblib(bundle_path) if bundle_path.exists() else None
        predictions_path = self.current_dir / "stage2_holdout_predictions.parquet"
        predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame(columns=["member_id"])
        if "member_id" in predictions.columns:
            predictions["member_id"] = predictions["member_id"].astype(str).str.strip().str.upper()
            predictions = predictions.set_index("member_id", drop=False)
        partnership_table_file = str(manifest.get("partnership_table_file", "partnership_table.parquet"))
        partnership_table_path = self.current_dir / partnership_table_file
        partnership_table = pd.read_parquet(partnership_table_path) if partnership_table_path.exists() else pd.DataFrame()
        ccs_table_file = str(manifest.get("ccs_concentration_table_file", "ccs_concentration_table.parquet"))
        ccs_table_path = self.current_dir / ccs_table_file
        ccs_table = pd.read_parquet(ccs_table_path) if ccs_table_path.exists() else pd.DataFrame()
        return ArtifactBundle(
            stage2_holdout_predictions_df=predictions,
            serving_manifest=manifest,
            snapshot_metadata={},
            promotion_metadata=promotion_metadata,
            evaluation_metadata=evaluation,
            run_metadata=run_metadata,
            snapshot_available=bool(predictions_path.exists()),
            snapshot_reason=None if predictions_path.exists() else "No partnership holdout predictions are available.",
            loaded_at=datetime.now(timezone.utc),
            source_run_id=str(manifest.get("run_id")),
            promoted_at=manifest.get("promoted_at"),
            evaluated_at=evaluation.get("evaluated_at"),
            model_version=str(manifest.get("model_version", self.default_model_version)),
            model_bundle=model_bundle,
            partnership_table_df=partnership_table,
            ccs_concentration_table_df=ccs_table,
            training_raw_parquet_path=run_dir / "data_ingestion" / "raw_data.parquet",
        )
