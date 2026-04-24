from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from fraud_detection.constants.constants import REPO_ROOT


_weekly_module = None


def _load_weekly_module():
    global _weekly_module
    if _weekly_module is not None:
        return _weekly_module

    import sys

    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _weekly_module = import_module("hybrid_inference")
    return _weekly_module


def get_weekly_lookback_days() -> int:
    module = _load_weekly_module()
    return int(module.WEEKLY_LOOKBACK_DAYS)


def clear_weekly_scored_cohort_cache() -> None:
    module = _load_weekly_module()
    module.clear_weekly_scored_cohort_cache()


def score_member_id(member_id: str, **kwargs: Any):
    module = _load_weekly_module()
    return module.score_member_id(member_id, **kwargs)


def get_weekly_member_score(member_id: str, **kwargs: Any):
    module = _load_weekly_module()
    return module.get_weekly_member_score(member_id, **kwargs)


def load_weekly_scored_cohort(**kwargs: Any):
    module = _load_weekly_module()
    return module.load_weekly_scored_cohort(**kwargs)


def get_inference_errors() -> tuple[type[Exception], type[Exception]]:
    module = _load_weekly_module()
    return module.MemberNotFoundError, module.InsufficientHistoryError
