"""
Unit tests for output directory organization and artifact preservation.
Verifies that all checkpoints, evaluation reports, training curves, confusion matrices,
and prediction artifacts for the single CNN + LSTM model are properly organized and validated.
"""

import json
from pathlib import Path
import pytest

from src.utils.output_organizer import (
    ensure_output_directories,
    organize_and_save_artifacts,
    verify_outputs_integrity,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_standard_output_directories_exist():
    """Verify that all required output directories exist on disk."""
    dirs = ensure_output_directories(PROJECT_ROOT)
    for dir_name, dir_path in dirs.items():
        assert dir_path.is_dir(), f"Expected directory does not exist: {dir_path}"


def test_single_cnn_lstm_checkpoints_exist_and_match():
    """Verify that checkpoints exist in outputs/checkpoints and match outputs/models."""
    models_dir = PROJECT_ROOT / "outputs" / "models"
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"

    for weight_file in ["cnn_lstm_head.keras", "cnn_lstm_end_to_end.keras"]:
        model_path = models_dir / weight_file
        ckpt_path = checkpoints_dir / weight_file

        assert model_path.is_file(), f"Missing model file: {model_path}"
        assert ckpt_path.is_file(), f"Missing checkpoint file: {ckpt_path}"
        assert model_path.stat().st_size > 0
        assert ckpt_path.stat().st_size == model_path.stat().st_size


def test_single_cnn_lstm_evaluation_report_schema_and_values():
    """Verify that the evaluation report contains correct metrics for the CNN + LSTM model."""
    report_path = PROJECT_ROOT / "outputs" / "reports" / "cnn_lstm_evaluation_report.json"
    assert report_path.is_file(), f"Missing evaluation report: {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_architecture"] == "CNN_LSTM"
    assert data["num_classes"] == 101
    assert data["sequence_length"] == 16
    assert data["observation_ratio"] == 0.5
    assert data["held_out_test_metrics"]["test_accuracy_pct"] == 68.98
    assert data["held_out_test_metrics"]["top5_accuracy_pct"] == 82.51
    assert round(data["held_out_test_metrics"]["test_loss"], 4) == 1.4993
    assert data["validation_metrics"]["val_accuracy_pct"] == 67.66


def test_single_cnn_lstm_plots_exist():
    """Verify that training curve and confusion matrix plots exist and are non-empty."""
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    expected_plots = [
        plots_dir / "cnn_lstm_training_curves.png",
        plots_dir / "cnn_lstm_val_confusion_matrix.png",
        plots_dir / "cnn_lstm_test_confusion_matrix.png",
    ]
    for plot_path in expected_plots:
        assert plot_path.is_file(), f"Missing plot: {plot_path}"
        assert plot_path.stat().st_size > 1000, f"Plot file seems truncated: {plot_path}"


def test_prediction_artifacts_exist():
    """Verify that prediction JSON and PNG visualizations exist for unseen test videos."""
    preds_dir = PROJECT_ROOT / "outputs" / "predictions"
    expected_preds = [
        preds_dir / "basketball_lstm_prediction.json",
        preds_dir / "basketball_lstm_prediction.png",
        preds_dir / "biking_lstm_prediction.json",
        preds_dir / "biking_lstm_prediction.png",
    ]
    for pred_path in expected_preds:
        assert pred_path.is_file(), f"Missing prediction artifact: {pred_path}"
        assert pred_path.stat().st_size > 0


def test_verify_outputs_integrity_all_true():
    """Verify that all output integrity checks pass."""
    results = verify_outputs_integrity(PROJECT_ROOT)
    for check_name, status in results.items():
        assert status is True, f"Integrity check failed for: {check_name}"


def test_gitignore_contains_critical_output_patterns():
    """Verify that .gitignore contains necessary patterns to ignore large binaries."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")
    for pattern in ["outputs/checkpoints/", "outputs/models/", "outputs/cache/", "outputs/uploads/"]:
        assert pattern in content, f"Missing pattern in .gitignore: {pattern}"
