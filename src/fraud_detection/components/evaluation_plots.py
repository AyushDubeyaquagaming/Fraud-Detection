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
        "member_score_scatter_rule_vs_ml.png",
        output_dir,
        _plot_rule_vs_ml_scatter,
        scored,
        labels,
    )
    _safe_plot(summary, "member_feature_space_pca.png", output_dir, _plot_feature_pca, scored, labels, feature_columns)
    _safe_plot(
        summary,
        "feature_importance.png",
        output_dir,
        _plot_feature_importance,
        scored,
        stage2_model_path,
        feature_columns,
    )

    if label_status != "available":
        for name in [
            "confusion_matrix.png",
            "pr_curve.png",
            "score_distribution.png",
            "capture_curve.png",
        ]:
            summary["skipped"].append({"plot": name, "reason": "label_status_unavailable"})
        logger.info("Skipping label-dependent evaluation plots because no positive reviewed labels are available.")
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


def _numeric_feature_frame(scored: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    frame = scored[[column for column in feature_columns if column in scored.columns]].apply(pd.to_numeric, errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _select_plot_features(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    priority = [
        "max_stage1_score",
        "mean_stage1_score",
        "max_strict_pair_score_today",
        "n_strict_pairs_in_draw",
        "clique_size_estimate",
        "clique_total_stake_share",
        "clique_max_pair_rule_confidence",
        "n_distinct_high_score_partners",
        "best_partnership_union_coverage",
        "pct_draws_in_persistent_partnership",
        "ccs_profit_share_1d",
        "ccs_profit_share_7d",
        "ccs_high_concentration_1d",
        "ccs_high_concentration_7d",
    ]
    selected = [column for column in priority if column in frame.columns]
    remaining = [column for column in frame.columns if column not in selected]
    variability = frame[remaining].std(skipna=True).sort_values(ascending=False) if remaining else pd.Series(dtype=float)
    selected.extend([column for column in variability.index if column not in selected])
    return frame[selected[:limit]]


def _stage1_signal(scored: pd.DataFrame) -> pd.Series:
    candidates = [
        "max_stage1_score",
        "mean_stage1_score",
        "max_strict_pair_score_today",
        "clique_max_pair_rule_confidence",
    ]
    series = [
        pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
        for column in candidates
        if column in scored.columns
    ]
    if not series:
        return pd.Series(0.0, index=scored.index)
    return pd.concat(series, axis=1).max(axis=1).clip(0, 1)


def _rule_signal(scored: pd.DataFrame) -> pd.Series:
    candidates = [
        "clique_max_pair_rule_confidence",
        "max_strict_pair_score_today",
        "max_stage1_score",
        "mean_stage1_score",
    ]
    series = [
        pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
        for column in candidates
        if column in scored.columns
    ]
    if "n_strict_pairs_in_draw" in scored.columns:
        series.append(pd.to_numeric(scored["n_strict_pairs_in_draw"], errors="coerce").fillna(0.0).clip(0, 1))
    if not series:
        return pd.Series(0.0, index=scored.index)
    return pd.concat(series, axis=1).max(axis=1).clip(0, 1)


def _jitter(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0, 1)
    if values.nunique(dropna=True) > max(8, int(len(values) * 0.02)):
        return values
    rng = np.random.default_rng(42)
    jittered = values.to_numpy(dtype=float) + rng.normal(0.0, 0.004, size=len(values))
    return pd.Series(np.clip(jittered, 0.0, 1.0), index=values.index)


def _feature_signal_fallback(scored: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    frame = _numeric_feature_frame(scored, feature_columns)
    frame = frame.dropna(axis=1, how="all")
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    if frame.empty:
        return {}
    target = _scores(scored)
    if target.nunique(dropna=True) <= 1:
        target = _stage1_signal(scored)
    values: pd.Series
    if target.nunique(dropna=True) > 1:
        values = frame.corrwith(target, method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        median = frame.median(skipna=True)
        mad = frame.sub(median, axis=1).abs().median(skipna=True)
        values = mad.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if float(values.max() or 0.0) > 0:
            values = values / float(values.max())
    values = values.loc[values.abs().sort_values(ascending=False).index]
    return {column: float(values.loc[column]) for column in values.index[:25] if np.isfinite(values.loc[column])}


def _plot_correlation(target: Path, scored: pd.DataFrame, feature_columns: list[str]) -> None:
    plt, sns = _setup_matplotlib()
    if not feature_columns:
        raise ValueError("no_stage2_features_present")
    frame = _numeric_feature_frame(scored, feature_columns)
    frame = frame.dropna(axis=1, how="all")
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    if frame.empty:
        raise ValueError("no_nonconstant_stage2_features_present")
    frame = _select_plot_features(frame, limit=18)
    corr = frame.corr(method="spearman").fillna(0.0)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    height = max(5, min(12, len(frame.columns) * 0.52))
    fig, ax = plt.subplots(figsize=(max(7, height * 1.15), height))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=len(frame.columns) <= 12,
        fmt=".2f",
        linewidths=0.25,
        linecolor="#f0f0f0",
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Spearman rho"},
    )
    ax.set_title("Stage 2 Feature Correlation (Top Nonconstant Signals)")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_feature_importance(
    target: Path,
    scored: pd.DataFrame,
    stage2_model_path: Path | None,
    feature_columns: list[str],
) -> None:
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
        feature_importances = getattr(estimator, "feature_importances_", None)
        if coef is not None and len(labels):
            values = np.ravel(coef)[: len(labels)]
        elif feature_importances is not None and len(labels):
            values = np.ravel(feature_importances)[: len(labels)]
        else:
            note = f"Model importances unavailable for {type(estimator).__name__}."
    if len(values) and not np.any(np.abs(values) > 0):
        fallback = _feature_signal_fallback(scored, labels)
        if fallback:
            labels = list(fallback.keys())
            values = np.array(list(fallback.values()), dtype=float)
            if note is None:
                note = "Stage 2 model has no usable coefficients; showing score-aligned feature signal."
        elif note is None:
            note = "Stage 2 model importances are all zero."
    order = np.argsort(np.abs(values))[-min(25, len(values)) :] if len(values) else []
    fig, ax = plt.subplots(figsize=(9, max(4, len(order) * 0.35)))
    if len(order):
        colors = ["#2f7d32" if values[i] >= 0 else "#b23b3b" for i in order]
        ax.barh([labels[i] for i in order], [values[i] for i in order], color=colors)
        ax.axvline(0, color="#222222", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "No Stage 2 feature columns available.", ha="center", va="center")
    if note:
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_title("Stage 2 Feature Importance")
    ax.set_xlabel("Model coefficient/importances, or fallback signal when unavailable")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _plot_feature_pca(target: Path, scored: pd.DataFrame, labels: pd.Series, feature_columns: list[str]) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plt, _ = _setup_matplotlib()
    frame = scored[feature_columns].apply(pd.to_numeric, errors="coerce") if feature_columns else pd.DataFrame(index=scored.index)
    frame = frame.dropna(axis=1, how="all").fillna(0.0)
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("pca_requires_at_least_two_rows_and_features")
    embedding = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(frame))
    scores = _scores(scored)
    top_k = max(1, int(len(scored) * 0.05)) if len(scored) else 0
    top_idx = set(scores.nlargest(top_k).index) if top_k else set()
    colors = [
        "#d62728" if labels.loc[idx] == 1 else "#f0a202" if idx in top_idx else "#1f77b4"
        for idx in scored.index
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=12, alpha=0.65, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Stage 2 Feature PCA")
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


def _plot_rule_vs_ml_scatter(target: Path, scored: pd.DataFrame, labels: pd.Series) -> None:
    plt, _ = _setup_matplotlib()
    rule_score = _rule_signal(scored)
    ml_score = _scores(scored)
    y_score = ml_score
    y_label = "Stage 2 ML score"
    if ml_score.nunique(dropna=True) <= 1:
        stage1_signal = _stage1_signal(scored)
        if stage1_signal.nunique(dropna=True) > 1:
            y_score = stage1_signal
            y_label = "Stage 1/member signal (Stage 2 constant)"
        else:
            y_score = rule_score
            y_label = "Rule signal (Stage 2 constant)"
    top_k = max(1, int(len(scored) * 0.05)) if len(scored) else 0
    top_idx = set(y_score.nlargest(top_k).index) if top_k else set()
    colors = [
        "#d62728" if labels.loc[idx] == 1 else "#f0a202" if rule_score.loc[idx] >= 0.5 or idx in top_idx else "#1f77b4"
        for idx in scored.index
    ]
    counts = {
        "confirmed": int(labels.sum()),
        "suspected": int(sum(color == "#f0a202" for color in colors)),
        "other": int(sum(color == "#1f77b4" for color in colors)),
    }
    agreement = float(((rule_score >= 0.5) == (y_score >= 0.5)).mean()) if len(scored) else 0.0
    plot_x = _jitter(rule_score.clip(0, 1))
    plot_y = _jitter(y_score.clip(0, 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_x, plot_y, c=colors, s=14, alpha=0.68, linewidths=0)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Rule confidence")
    ax.set_ylabel(y_label)
    ax.set_title(
        "Rule vs Member Signal\n"
        f"confirmed={counts['confirmed']}  suspected={counts['suspected']}  other={counts['other']}  agreement={agreement:.2f}",
        fontsize=12,
    )
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="confirmed fraud", markerfacecolor="#d62728", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="suspected", markerfacecolor="#f0a202", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="other", markerfacecolor="#1f77b4", markersize=7),
        ],
        loc="best",
        frameon=False,
    )
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
    y_label = "Stage 2 score"
    if y.nunique(dropna=True) <= 1:
        y_label = "Stage 2 score (constant, jittered)"
    top_k = max(1, int(len(scored) * 0.05)) if len(scored) else 0
    top_idx = set(y.nlargest(top_k).index) if top_k else set()
    colors = [
        "#d62728" if labels.loc[idx] == 1 else "#f0c419" if idx in top_idx else "#1f77b4"
        for idx in scored.index
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(_jitter(x.clip(0, 1)), _jitter(y.clip(0, 1)), c=colors, s=12, alpha=0.65, linewidths=0)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="confirmed fraud", markerfacecolor="#d62728", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="top 5pct score", markerfacecolor="#f0c419", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="other", markerfacecolor="#1f77b4", markersize=7),
        ],
        loc="best",
        frameon=False,
    )
    ax.set_xlabel("Max member Stage 1 score")
    ax.set_ylabel(y_label)
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
