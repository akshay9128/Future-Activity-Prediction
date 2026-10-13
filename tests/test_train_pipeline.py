"""
Test Suite for Unified Training Pipeline and Model Benchmarking.

Validates summary aggregation, comparative metrics consistency, artifact generation,
and architectural parameter hierarchy across all trained models.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pytest

from src.models.train_pipeline import (
    generate_comparison_summary,
    get_model_parameters,
    plot_model_comparison,
    run_training_pipeline,
)


def test_generate_comparison_summary_structure():
    """Verify that generate_comparison_summary parses all model metrics into standard schema."""
    summary = generate_comparison_summary()
    assert "models" in summary, "Summary must contain 'models' key"
    assert len(summary["models"]) == 3, f"Expected 3 model entries, got {len(summary['models'])}"

    expected_keys = {
        "model_key",
        "display_name",
        "val_loss",
        "val_accuracy",
        "val_accuracy_pct",
        "macro_f1",
        "trainable_parameters",
        "epochs_trained",
        "val_samples_total",
    }

    for record in summary["models"]:
        missing = expected_keys - set(record.keys())
        assert not missing, f"Record {record.get('model_key')} missing keys: {missing}"
        assert 0.0 <= record["val_accuracy"] <= 1.0, f"Invalid accuracy: {record['val_accuracy']}"
        assert 0.0 <= record["val_accuracy_pct"] <= 100.0, f"Invalid accuracy %: {record['val_accuracy_pct']}"
        assert record["val_loss"] > 0.0, f"Invalid val_loss: {record['val_loss']}"
        assert record["trainable_parameters"] > 0, f"Invalid parameter count: {record['trainable_parameters']}"
        assert record["val_samples_total"] in (99, 303, 1998), f"Unexpected validation sample count: {record['val_samples_total']}"


def test_comparison_summary_json_file_created():
    """Verify that model_comparison_val.json exists on disk and is readable."""
    json_path = Path("outputs/logs/model_comparison_val.json")
    assert json_path.exists(), f"File does not exist: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    assert len(data["models"]) == 3


def test_comparison_chart_created():
    """Verify that model_comparison_summary.png exists and is a valid non-empty plot file."""
    chart_path = Path("outputs/logs/model_comparison_summary.png")
    assert chart_path.exists(), f"Plot does not exist: {chart_path}"
    assert chart_path.stat().st_size > 10_000, "Plot file size suspiciously small"


def test_run_training_pipeline_skip_training():
    """Verify pipeline orchestration with skip_training=True executes cleanly."""
    result = run_training_pipeline(skip_training=True)
    assert isinstance(result, dict)
    assert "models" in result
    assert len(result["models"]) == 3


def test_model_parameter_hierarchy():
    """
    Verify architectural parameter hierarchy:
    CNN Baseline < CNN + LSTM < CNN + LSTM + Attention
    """
    summary = generate_comparison_summary()
    param_map = {m["model_key"]: m["trainable_parameters"] for m in summary["models"]}

    baseline_params = param_map["CNN_Only_Baseline"]
    lstm_params = param_map["CNN_LSTM"]
    attention_params = param_map["CNN_LSTM_Attention"]

    assert baseline_params < lstm_params, (
        f"CNN Baseline ({baseline_params}) should have fewer parameters than CNN+LSTM ({lstm_params})"
    )
    assert lstm_params < attention_params, (
        f"CNN+LSTM ({lstm_params}) should have fewer parameters than CNN+LSTM+Attention ({attention_params})"
    )
