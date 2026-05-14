from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraud_detection.logger import get_logger

logger = get_logger(__name__)


def generate_evaluation_plots(
    *,
    scored: pd.DataFrame,
    output_dir: Path,
    training_report: dict[str, Any],
    stage2_model_path: Path | None,
    stage1_oof_predictions_path: Path | None = None,
    capture_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate MLflow-friendly evaluation PNGs for the Stage 2 evaluation frame.

    Plot generation is deliberately best-effort per plot: a bad diagnostic should
    not invalidate a trained/promoted artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = _actual_stage2_feature_columns(scored, training_report)
    labels = _labels(scored)
    label_status = "available" if int(labels.sum()) > 0 else "unavailable"
    summary: dict[str, Any] = {"generated": [], "skipped": [], "label_status": label_status}

    _safe_plot(summary, "feature_correlation_heatmap.png", output_dir, _plot_correlation, scored, feature_columns)
    _safe_plot(
        summary,
        "feature_importance.png",
        output_dir,
        _plot_feature_importance,
        stage2_model_path,
        feature_columns,
    )

    if label_status != "available":
        for name in [
            "confusion_matrix.png",
            "score_scatter_stage1_vs_stage2.png",
            "pr_curve.png",
            "score_distribution.png",
            "capture_curve.png",
        ]:
            summary["skipped"].append({"plot": name, "reason": "label_status_unavailable"})
        logger.info("Skipping label-dependent evaluation plots because no positive analyst labels are available.")
    else:
        threshold = float(training_report.get("stage2_alert_threshold", 0.65) or 0.65)
        _safe_plot(summary, "confusion_matrix.png", output_dir, _plot_confusion_matrix, scored, labels, threshold)
        _safe_plot(
            summary,
            "score_scatter_stage1_vs_stage2.png",
            output_dir,
            _plot_stage1_stage2_scatter,
            scored,
            labels,
            stage1_oof_predictions_path,
        )
        _safe_plot(summary, "pr_curve.png", output_dir, _plot_pr_curve, scored, labels)
        _safe_plot(summary, "score_distribution.png", output_dir, _plot_score_distribution, scored, labels)
        _safe_plot(summary, "capture_curve.png", output_dir, _plot_capture_curve, scored, labels, capture_stats or {})

    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _safe_plot(summary: dict[str, Any], name: str, output_dir: Path, func, *args) -> None:
    target = output_dir / name
    try:
        func(target, *args)
        summary["generated"].append(name)
    except Exception as exc:
        logger.warning("Evaluation plot %s skipped: %s", name, exc)
        summary["skipped"].append({"plot": name, "reason": str(exc)})


def _actual_stage2_feature_columns(scored: pd.DataFrame, training_report: dict[str, Any]) -> list[str]:
    configured = list(training_report.get("stage2_feature_columns") or [])
    if configured:
        return [column for column in configured if column in scored.columns]
    excluded = {"member_id", "label_gold_member", "stage2_score", "sample_weight"}
    return [
        column
        for column in scored.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(scored[column])
    ]


def _labels(scored: pd.DataFrame) -> pd.Series:
    if "label_gold_member" not in scored.columns:
        return pd.Series(0, index=scored.index, dtype=int)
    return pd.to_numeric(scored["label_gold_member"], errors="coerce").fillna(0).astype(int)


def _scores(scored: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(scored.get("stage2_score", 0.0), errors="coerce").fillna(0.0)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _plot_correlation(target: Path, scored: pd.DataFrame, feature_columns: list[str]) -> None:
    plt, sns = _setup_matplotlib()
    if not feature_columns:
        raise ValueError("no_stage2_features_present")
    frame = scored[feature_columns].apply(pd.to_numeric, errors="coerce")
    corr = frame.corr(method="pearson").fillna(0.0)
    height = max(6, min(18, len(feature_columns) * 0.45))
    fig, ax = plt.subplots(figsize=(max(8, height * 1.2), height))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Stage 2 Feature Correlation")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_feature_importance(target: Path, stage2_model_path: Path | None, feature_columns: list[str]) -> None:
    plt, _ = _setup_matplotlib()
    labels = list(feature_columns)
    values = np.zeros(len(labels), dtype=float)
    note = None
    if stage2_model_path is None or not Path(stage2_model_path).exists():
        note = "Stage 2 model artifact unavailable."
    else:
        model = joblib.load(stage2_model_path)
        estimator = getattr(model, "named_steps", {}).get("model") if hasattr(model, "named_steps") else model
        coef = getattr(estimator, "coef_", None)
        if coef is not None and len(labels):
            values = np.ravel(coef)[: len(labels)]
        else:
            note = f"Coefficient importances unavailable for {type(estimator).__name__}."
    order = np.argsort(np.abs(values))[-min(25, len(values)) :] if len(values) else []
    fig, ax = plt.subplots(figsize=(9, max(4, len(order) * 0.35)))
    if len(order):
        ax.barh([labels[i] for i in order], [values[i] for i in order], color="#3b6ea8")
        ax.axvline(0, color="#222222", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "No Stage 2 feature columns available.", ha="center", va="center")
    if note:
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_title("Stage 2 Feature Importance")
    ax.set_xlabel("Standardized logistic coefficient")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(target: Path, scored: pd.DataFrame, labels: pd.Series, threshold: float) -> None:
    plt, sns = _setup_matplotlib()
    predictions = (_scores(scored) >= threshold).astype(int)
    matrix = pd.crosstab(labels, predictions, rownames=["Actual"], colnames=["Predicted"], dropna=False)
    matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    row_pct = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    annotations = matrix.astype(str) + "\n" + row_pct.round(1).astype(str) + "%"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix at Stage 2 >= {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_stage1_stage2_scatter(
    target: Path,
    scored: pd.DataFrame,
    labels: pd.Series,
    stage1_oof_predictions_path: Path | None,
) -> None:
    plt, _ = _setup_matplotlib()
    x = _stage1_member_scores(scored, stage1_oof_predictions_path)
    y = _scores(scored)
    top_k = max(1, int(len(scored) * 0.05)) if len(scored) else 0
    top_idx = set(y.nlargest(top_k).index) if top_k else set()
    colors = [
        "#d62728" if labels.loc[idx] == 1 else "#f0c419" if idx in top_idx else "#1f77b4"
        for idx in scored.index
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=colors, s=12, alpha=0.65, linewidths=0)
    ax.set_xlabel("Max member Stage 1 score")
    ax.set_ylabel("Stage 2 score")
    ax.set_title("Stage 1 vs Stage 2 Scores")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _stage1_member_scores(scored: pd.DataFrame, stage1_oof_predictions_path: Path | None) -> pd.Series:
    if "max_stage1_score" in scored.columns:
        return pd.to_numeric(scored["max_stage1_score"], errors="coerce").fillna(0.0)
    if stage1_oof_predictions_path and Path(stage1_oof_predictions_path).exists() and "member_id" in scored.columns:
        stage1 = pd.read_parquet(stage1_oof_predictions_path)
        if {"member_id", "stage1_score"}.issubset(stage1.columns):
            lookup = (
                stage1.assign(member_id=stage1["member_id"].astype(str).str.strip().str.upper())
                .groupby("member_id")["stage1_score"]
                .max()
            )
            return scored["member_id"].astype(str).str.strip().str.upper().map(lookup).fillna(0.0)
    return pd.Series(0.0, index=scored.index)


def _plot_pr_curve(target: Path, scored: pd.DataFrame, labels: pd.Series) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt, _ = _setup_matplotlib()
    scores = _scores(scored)
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#3b6ea8")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP={ap:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_score_distribution(target: Path, scored: pd.DataFrame, labels: pd.Series) -> None:
    plt, sns = _setup_matplotlib()
    frame = pd.DataFrame({"stage2_score": _scores(scored), "label": labels.map({1: "confirmed_fraud", 0: "non_fraud"})})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(frame, x="stage2_score", hue="label", bins=40, stat="count", common_norm=False, ax=ax)
    ax.set_yscale("symlog")
    ax.set_title("Stage 2 Score Distribution")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_capture_curve(target: Path, scored: pd.DataFrame, labels: pd.Series, capture_stats: dict[str, Any]) -> None:
    plt, _ = _setup_matplotlib()
    scores = _scores(scored)
    ordered = labels.loc[scores.sort_values(ascending=False).index]
    total = int(labels.sum())
    percentages = np.arange(1, 101)
    captures = []
    for pct in percentages:
        k = max(1, int(len(ordered) * pct / 100.0))
        captures.append(float(ordered.iloc[:k].sum() / total) if total else 0.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(percentages, captures, color="#3b6ea8")
    ax.axvline(5, color="#d62728", linestyle="--", linewidth=1)
    top5 = (capture_stats.get("top_5pct") or {}).get("capture_rate")
    if top5 is not None:
        ax.scatter([5], [float(top5)], color="#d62728", zorder=3)
    ax.set_xlabel("Top population percentage")
    ax.set_ylabel("Fraud capture rate")
    ax.set_title("Capture Curve")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
