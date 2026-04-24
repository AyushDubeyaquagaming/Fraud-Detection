from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.constants.constants import (
    HYBRID_EVALUATION_FILE,
    HYBRID_SCORED_PLAYERS_FILE,
    PROMOTION_METADATA_FILE,
    REPO_ROOT,
    RUN_METADATA_FILE,
    WEEKLY_SCORING_MANIFEST_FILE,
)
from fraud_detection.utils.common import load_parquet, read_json


@dataclass(frozen=True)
class ArtifactBundle:
    scored_players_df: pd.DataFrame
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
        snapshot_manifest_path = self.current_dir / WEEKLY_SCORING_MANIFEST_FILE
        snapshot_metadata = read_json(snapshot_manifest_path) if snapshot_manifest_path.exists() else {}
        snapshot_scored_path = self.current_dir / snapshot_metadata.get("scored_players_file", HYBRID_SCORED_PLAYERS_FILE)
        snapshot_eval_path = self.current_dir / snapshot_metadata.get("evaluation_file", HYBRID_EVALUATION_FILE)
        snapshot_available = bool(snapshot_metadata) and snapshot_scored_path.exists() and snapshot_eval_path.exists()

        if not snapshot_metadata:
            snapshot_reason = "Weekly serving snapshot has not been generated yet."
        elif not snapshot_available:
            snapshot_reason = "Weekly serving snapshot artifacts are incomplete."
        else:
            snapshot_reason = None

        run_meta_path = run_dir / RUN_METADATA_FILE
        promotion_meta_path = self.current_dir / PROMOTION_METADATA_FILE

        run_metadata = read_json(run_meta_path)
        promotion_metadata = read_json(promotion_meta_path) if promotion_meta_path.exists() else {}

        if snapshot_available:
            df = load_parquet(snapshot_scored_path)
            df["member_id"] = df["member_id"].astype(str).str.strip().str.upper()
            df = df.set_index("member_id", drop=False)
            evaluation_metadata = read_json(snapshot_eval_path)
        else:
            df = pd.DataFrame(columns=["member_id"]).set_index("member_id", drop=False)
            evaluation_metadata = {
                "scored_at": None,
                "total_players": 0,
                "risk_tier_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
                "anomaly_weight": 0.6,
                "supervised_weight": 0.4,
                "snapshot_status": "insufficient_data",
                "snapshot_reason": snapshot_reason,
            }

        snapshot_metadata = {
            **snapshot_metadata,
            "snapshot_available": snapshot_available,
            "snapshot_status": "ready" if snapshot_available else "insufficient_data",
            "snapshot_reason": snapshot_reason,
        }

        return ArtifactBundle(
            scored_players_df=df,
            serving_manifest=manifest,
            snapshot_metadata=snapshot_metadata,
            promotion_metadata=promotion_metadata,
            evaluation_metadata=evaluation_metadata,
            run_metadata=run_metadata,
            snapshot_available=snapshot_available,
            snapshot_reason=snapshot_reason,
            loaded_at=datetime.now(timezone.utc),
            source_run_id=str(snapshot_metadata.get("source_run_id") or manifest["run_id"]),
            promoted_at=manifest.get("promoted_at"),
            evaluated_at=evaluation_metadata.get("scored_at") or evaluation_metadata.get("evaluated_at"),
            model_version=str(
                snapshot_metadata.get("model_version")
                or manifest.get("model_version", self.default_model_version)
            ),
        )
