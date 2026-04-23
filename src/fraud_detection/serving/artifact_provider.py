from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.constants.constants import (
    EVALUATION_REPORT_FILE,
    PROMOTION_METADATA_FILE,
    REPO_ROOT,
    RUN_METADATA_FILE,
    SCORED_PLAYERS_FILE,
)
from fraud_detection.utils.common import load_parquet, read_json


@dataclass(frozen=True)
class ArtifactBundle:
    scored_players_df: pd.DataFrame
    serving_manifest: dict[str, Any]
    promotion_metadata: dict[str, Any]
    evaluation_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    loaded_at: datetime
    source_run_id: str
    promoted_at: str | None
    evaluated_at: str | None
    model_version: str


class ArtifactProvider(ABC):
    @abstractmethod
    def load(self) -> ArtifactBundle:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class LocalDiskArtifactProvider(ArtifactProvider):
    def __init__(
        self,
        current_dir: Path,
        manifest_file: str = "serving_manifest.json",
        default_model_version: str = "hybrid_v1",
        repo_root: Path = REPO_ROOT,
    ):
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

        scored_path = run_dir / "model_evaluation" / SCORED_PLAYERS_FILE
        eval_path = run_dir / "model_evaluation" / EVALUATION_REPORT_FILE
        run_meta_path = run_dir / RUN_METADATA_FILE
        promotion_meta_path = self.current_dir / PROMOTION_METADATA_FILE

        df = load_parquet(scored_path)
        df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
        df = df.set_index("member_id", drop=False)

        evaluation_metadata = read_json(eval_path)
        run_metadata = read_json(run_meta_path)
        promotion_metadata = read_json(promotion_meta_path) if promotion_meta_path.exists() else {}

        return ArtifactBundle(
            scored_players_df=df,
            serving_manifest=manifest,
            promotion_metadata=promotion_metadata,
            evaluation_metadata=evaluation_metadata,
            run_metadata=run_metadata,
            loaded_at=datetime.now(timezone.utc),
            source_run_id=manifest["run_id"],
            promoted_at=manifest.get("promoted_at"),
            evaluated_at=evaluation_metadata.get("evaluated_at"),
            model_version=manifest.get("model_version", self.default_model_version),
        )
