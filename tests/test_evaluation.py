"""
Test Suite for Held-Out Test Set Evaluation and Generalization Metrics.

Validates that test features are cached, evaluation runs correctly, all 3 models
are evaluated on 99 test videos, confusion matrices are valid 5x5 matrices summing
to 99, and visualization plots are produced.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.models.evaluate_test_set import evaluate_single_model


def test_test_features_cache_exists_and_shape():
    """Verify that outputs/cache/test_features.npz exists and holds 99 feature sequences."""
    cache_path = Path("outputs/cache/test_features.npz")
    assert cache_path.exists(), f"Test feature cache missing: {cache_path}"

    data = np.load(cache_path)
    assert "features" in data, "Cached test file missing 'features' key"
    assert "labels" in data, "Cached test file missing 'labels' key"

    features = data["features"]
    labels = data["labels"]

    assert features.shape == (99, 16, 1280), f"Expected shape (99, 16, 1280), got {features.shape}"
    assert labels.shape == (99,), f"Expected labels shape (99,), got {labels.shape}"
    assert np.all(np.isfinite(features)), "Test features contain NaNs or Infinities"


def test_test_evaluation_artifacts_exist():
    """Verify that all output artifacts produced during test evaluation exist on disk."""
    artifacts = [
        Path("outputs/logs/test_set_evaluation.json"),
        Path("outputs/logs/test_confusion_matrices.png"),
        Path("outputs/logs/test_metrics_comparison.png"),
    ]
    for artifact in artifacts:
        assert artifact.exists(), f"Expected artifact {artifact} missing"
        assert artifact.stat().st_size > 1000, f"Artifact {artifact} is unexpectedly empty"


def test_test_evaluation_metrics_schema():
    """Verify schema, sample counts, and value constraints in test_set_evaluation.json."""
    json_path = Path("outputs/logs/test_set_evaluation.json")
    assert json_path.exists(), f"File missing: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("test_samples_total") == 99
    assert len(data.get("classes", [])) == 5
    assert len(data.get("models", [])) == 3

    for model_rec in data["models"]:
        assert "model_key" in model_rec
        assert "display_name" in model_rec
        assert "test_loss" in model_rec
        assert "test_accuracy" in model_rec
        assert "confusion_matrix" in model_rec
        assert "classification_report" in model_rec
        assert "generalization_gap_accuracy" in model_rec

        # Accuracy must be within valid probability range
        acc = model_rec["test_accuracy"]
        assert 0.0 <= acc <= 1.0, f"Invalid accuracy: {acc}"
        assert model_rec["test_loss"] > 0.0, f"Invalid test loss: {model_rec['test_loss']}"


def test_confusion_matrices_shape_and_sum():
    """Verify each model's test confusion matrix is 5x5 and sums to 99 test samples."""
    json_path = Path("outputs/logs/test_set_evaluation.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for model_rec in data["models"]:
        cm = np.array(model_rec["confusion_matrix"])
        assert cm.shape == (5, 5), f"Confusion matrix for {model_rec['display_name']} must be (5, 5)"
        assert np.sum(cm) == 99, f"Confusion matrix for {model_rec['display_name']} must sum to 99"
        assert np.all(cm >= 0), "Confusion matrix entries must be non-negative"


def test_evaluate_single_model_function():
    """Verify programmatic execution of evaluate_single_model on a saved model."""
    model_path = Path("outputs/models/cnn_baseline_head.keras")
    assert model_path.exists(), "CNN baseline head model artifact missing"

    data = np.load("outputs/cache/test_features.npz")
    x_test = data["features"][:10]  # Subset of 10 samples
    y_test = data["labels"][:10]
    classes = ["ApplyEyeMakeup", "Basketball", "Biking", "JumpingJack", "WalkingWithDog"]

    res = evaluate_single_model(model_path, x_test, y_test, classes)
    assert "test_accuracy" in res
    assert "test_loss" in res
    assert "confusion_matrix" in res
    assert np.sum(res["confusion_matrix"]) == 10
