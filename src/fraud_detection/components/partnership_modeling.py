from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.components.partnership_features import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class PartnershipModelResult:
    model: Any
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    feature_columns: list[str]


def train_stage1_oof(
    stage1_labeled: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    label_col: str = "label_stage1",
    sample_weight_col: str = "sample_weight",
    random_seed: int = 42,
    n_splits: int = 5,
) -> PartnershipModelResult:
    """Train Stage 1 and return true out-of-fold predictions.

    The final model is fitted on all rows after OOF scores are generated, so it
    is suitable for serving while Stage 2 can consume leakage-safe OOF scores.
    """
    features = feature_columns or STAGE1_FEATURE_COLUMNS
    df = _prepare_labeled_frame(stage1_labeled, features, label_col, sample_weight_col)
    base_cols = [
        c
        for c in ["member_id", "draw_id", "draw_date", "best_partner_member_id", "member_a", "member_b"]
        if c in df.columns
    ]
    predictions = df[base_cols].copy()
    predictions["stage1_score"] = 0.0
    predictions["is_oof"] = False

    y = df[label_col].astype(int)
    weights = df[sample_weight_col].astype(float) if sample_weight_col in df.columns else None
    final_model = _new_classifier(random_seed)
    if y.nunique() < 2:
        final_model = Pipeline([("model", DummyClassifier(strategy="constant", constant=int(y.iloc[0]) if len(y) else 0))])
        final_model.fit(df[features], y)
        return PartnershipModelResult(
            model=final_model,
            predictions=predictions,
            metrics={"pr_auc": None, "roc_auc": None, "note": "single_class"},
            feature_columns=features,
        )

    split_count = min(int(n_splits), int(y.sum()), int((1 - y).sum()))
    if split_count >= 2:
        splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=random_seed)
        for train_idx, val_idx in splitter.split(df[features], y):
            fold_model = _new_classifier(random_seed)
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["model__sample_weight"] = weights.iloc[train_idx].to_numpy()
            fold_model.fit(df.iloc[train_idx][features], y.iloc[train_idx], **fit_kwargs)
            predictions.loc[df.index[val_idx], "stage1_score"] = _predict_proba(
                fold_model,
                df.iloc[val_idx][features],
            )
            predictions.loc[df.index[val_idx], "is_oof"] = True
    else:
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["model__sample_weight"] = weights.to_numpy()
        final_model.fit(df[features], y, **fit_kwargs)
        predictions["stage1_score"] = _predict_proba(final_model, df[features])
        predictions["is_oof"] = False

    final_fit_kwargs = {}
    if weights is not None:
        final_fit_kwargs["model__sample_weight"] = weights.to_numpy()
    final_model.fit(df[features], y, **final_fit_kwargs)
    metrics = _classification_metrics(y.loc[predictions["is_oof"]], predictions.loc[predictions["is_oof"], "stage1_score"])
    metrics["oof_rows"] = int(predictions["is_oof"].sum())
    metrics["total_rows"] = int(len(predictions))
    return PartnershipModelResult(final_model, predictions, metrics, features)


def train_stage2_model(
    stage2_features: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    label_col: str = "label_gold_member",
    random_seed: int = 42,
    n_splits: int = 5,
) -> PartnershipModelResult:
    features = [c for c in (feature_columns or STAGE2_FEATURE_COLUMNS) if c in stage2_features.columns]
    df = _prepare_labeled_frame(stage2_features, features, label_col, "sample_weight")
    predictions = df[["member_id"]].copy() if "member_id" in df.columns else pd.DataFrame(index=df.index)
    predictions["stage2_score"] = 0.0
    predictions["is_oof"] = False
    predictions["stage2_prediction_source"] = "not_evaluated"
    y = df[label_col].astype(int)
    weights = df["sample_weight"].astype(float) if "sample_weight" in df.columns else None
    model = _new_classifier(random_seed)
    model_frame = df[features] if features else pd.DataFrame({"bias": np.zeros(len(df))}, index=df.index)

    if y.nunique() < 2 or not features:
        constant = int(y.iloc[0]) if len(y) else 0
        model = Pipeline([("model", DummyClassifier(strategy="constant", constant=int(y.iloc[0]) if len(y) else 0))])
        model.fit(model_frame, y)
        predictions["stage2_score"] = _predict_proba(model, model_frame)
        validation_status = "not_evaluated_no_labels" if int(y.sum()) == 0 else "insufficient_labels_for_oof"
        predictions["stage2_prediction_source"] = validation_status
        metrics = _classification_metrics(pd.Series(dtype=int), pd.Series(dtype=float))
        metrics.update(
            {
                "validation_status": validation_status,
                "stage2_prediction_source": validation_status,
                "oof_rows": 0,
                "total_rows": int(len(predictions)),
                "positive_rows": int(y.sum()),
            }
        )
        return PartnershipModelResult(model, predictions, metrics, features)

    split_count = min(int(n_splits), int(y.sum()), int((1 - y).sum()))
    if split_count >= 2:
        splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=random_seed)
        for train_idx, val_idx in splitter.split(model_frame, y):
            fold_model = _new_classifier(random_seed)
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["model__sample_weight"] = weights.iloc[train_idx].to_numpy()
            fold_model.fit(model_frame.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
            predictions.loc[df.index[val_idx], "stage2_score"] = _predict_proba(
                fold_model,
                model_frame.iloc[val_idx],
            )
            predictions.loc[df.index[val_idx], "is_oof"] = True
        validation_status = "evaluated_oof"
        predictions["stage2_prediction_source"] = "oof"
    else:
        validation_status = "insufficient_labels_for_oof"
        predictions["stage2_prediction_source"] = validation_status

    final_fit_kwargs = {}
    if weights is not None:
        final_fit_kwargs["model__sample_weight"] = weights.to_numpy()
    model.fit(model_frame, y, **final_fit_kwargs)

    eval_mask = predictions["is_oof"].astype(bool)
    metrics = _classification_metrics(y.loc[eval_mask], predictions.loc[eval_mask, "stage2_score"])
    metrics.update(
        {
            "validation_status": validation_status,
            "stage2_prediction_source": "oof" if validation_status == "evaluated_oof" else validation_status,
            "oof_rows": int(eval_mask.sum()),
            "total_rows": int(len(predictions)),
            "positive_rows": int(y.sum()),
        }
    )
    return PartnershipModelResult(model, predictions, metrics, features)


def score_stage1(model: Any, stage1_features: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    features = feature_columns or STAGE1_FEATURE_COLUMNS
    out = stage1_features[
        [
            c
            for c in ["member_id", "draw_id", "draw_date", "best_partner_member_id", "member_a", "member_b"]
            if c in stage1_features.columns
        ]
    ].copy()
    out["stage1_score"] = _predict_proba(model, stage1_features[features])
    return out


def score_stage2(model: Any, stage2_features: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    features = [c for c in (feature_columns or STAGE2_FEATURE_COLUMNS) if c in stage2_features.columns]
    out = stage2_features[["member_id"]].copy() if "member_id" in stage2_features.columns else pd.DataFrame(index=stage2_features.index)
    out["stage2_score"] = _predict_proba(model, stage2_features[features])
    return out


def _new_classifier(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _prepare_labeled_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_col: str,
    sample_weight_col: str,
) -> pd.DataFrame:
    df = frame.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if label_col not in df.columns:
        df[label_col] = 0
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if sample_weight_col not in df.columns:
        df[sample_weight_col] = 1.0
    df[sample_weight_col] = pd.to_numeric(df[sample_weight_col], errors="coerce").fillna(1.0)
    return df


def _predict_proba(model: Any, frame: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(frame.replace([np.inf, -np.inf], np.nan))
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps.get("model"), "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(frame), dtype=float)


def _classification_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, Any]:
    if len(labels) == 0 or labels.nunique() < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(labels.astype(int), scores)),
        "roc_auc": float(roc_auc_score(labels.astype(int), scores)),
    }
