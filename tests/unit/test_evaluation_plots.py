from __future__ import annotations

import json

import pandas as pd

from fraud_detection.components.evaluation_plots import generate_evaluation_plots


def test_evaluation_plots_skip_label_dependent_outputs_without_positive_labels(tmp_path):
    scored = pd.DataFrame(
        {
            "member_id": ["A", "B", "C"],
            "stage2_score": [0.1, 0.2, 0.3],
            "label_gold_member": [0, 0, 0],
            "weekly_draw_count": [1, 2, 3],
            "max_stage1_score": [0.0, 0.5, 0.7],
        }
    )

    summary = generate_evaluation_plots(
        scored=scored,
        output_dir=tmp_path,
        training_report={"stage2_feature_columns": ["weekly_draw_count", "max_stage1_score"]},
        stage2_model_path=None,
    )

    assert (tmp_path / "feature_correlation_heatmap.png").exists()
    assert (tmp_path / "feature_importance.png").exists()
    assert not (tmp_path / "confusion_matrix.png").exists()
    assert summary["label_status"] == "unavailable"

    persisted = json.loads((tmp_path / "plot_summary.json").read_text(encoding="utf-8"))
    skipped = {entry["plot"] for entry in persisted["skipped"]}
    assert "pr_curve.png" in skipped
    assert "capture_curve.png" in skipped
