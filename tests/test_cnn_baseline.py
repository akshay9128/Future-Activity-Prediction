"""
 Test Suite: CNN-only baseline classifier and evaluation.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_baseline import (
    build_cnn_baseline_head,
    build_end_to_end_cnn_baseline,
)
from src.models.cnn_feature_extractor import build_cnn_feature_extractor


def test_cnn_baseline_head_output_shape():
    """Verify that head model takes (B, 16, 1280) and outputs (B, 5)."""
    model = build_cnn_baseline_head(sequence_length=16, feature_dim=1280, num_classes=5)
    dummy_input = np.random.uniform(-1.0, 1.0, size=(4, 16, 1280)).astype(np.float32)
    output = model(dummy_input, training=False)

    assert output.shape == (4, 5), f"Expected shape (4, 5), got {output.shape}"
    # Softmax probabilities must sum to 1.0 per sample
    prob_sums = np.sum(output.numpy(), axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
    assert np.all(output.numpy() >= 0.0), "Probabilities must be non-negative"


def test_cnn_baseline_training_step():
    """Verify that model trains for 2 epochs on synthetic data and loss remains finite."""
    model = build_cnn_baseline_head(sequence_length=16, feature_dim=1280, num_classes=5)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x_dummy = np.random.uniform(-1.0, 1.0, size=(16, 16, 1280)).astype(np.float32)
    y_dummy = np.random.randint(0, 5, size=(16,)).astype(np.int64)

    history = model.fit(x_dummy, y_dummy, epochs=2, batch_size=4, verbose=0)

    assert len(history.history["loss"]) == 2
    for loss_val in history.history["loss"]:
        assert np.isfinite(loss_val), "Loss value is not finite (NaN or Inf)!"
        assert loss_val > 0.0


def test_saved_model_reload_and_reproducibility():
    """Verify that saved .keras model reloads and reproduces exact predictions."""
    head_path = PROJECT_ROOT / "outputs" / "models" / "cnn_baseline_head.keras"
    assert head_path.is_file(), f"Model file missing: {head_path}"

    reloaded_model = tf.keras.models.load_model(head_path)
    dummy_input = np.ones((2, 16, 1280), dtype=np.float32)

    pred1 = reloaded_model.predict(dummy_input, verbose=0)
    pred2 = reloaded_model.predict(dummy_input, verbose=0)

    np.testing.assert_allclose(pred1, pred2, atol=1e-6)
    assert pred1.shape == (2, 5)


def test_saved_artifacts_exist():
    """Verify that  plots, models, and metric summaries exist."""
    expected = [
        PROJECT_ROOT / "outputs" / "models" / "cnn_baseline_head.keras",
        PROJECT_ROOT / "outputs" / "models" / "cnn_baseline_end_to_end.keras",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_baseline_training.png",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_baseline_confusion_matrix.png",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_baseline_val_metrics.json",
        PROJECT_ROOT / "outputs" / "cache" / "train_features.npz",
        PROJECT_ROOT / "outputs" / "cache" / "val_features.npz",
    ]
    for path in expected:
        assert path.is_file(), f"Expected artifact missing: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"


def test_end_to_end_model_prediction():
    """Verify end-to-end model accepts raw 5D video frames (1, 16, 224, 224, 3)."""
    e2e_path = PROJECT_ROOT / "outputs" / "models" / "cnn_baseline_end_to_end.keras"
    assert e2e_path.is_file(), f"End-to-end model missing: {e2e_path}"

    e2e_model = tf.keras.models.load_model(e2e_path)
    raw_video = np.random.uniform(-1.0, 1.0, size=(1, 16, 224, 224, 3)).astype(np.float32)
    probs = e2e_model.predict(raw_video, verbose=0)

    assert probs.shape == (1, 5)
    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-4)


if __name__ == "__main__":
    test_cnn_baseline_head_output_shape()
    test_cnn_baseline_training_step()
    test_saved_model_reload_and_reproducibility()
    test_saved_artifacts_exist()
    test_end_to_end_model_prediction()
    print("All CNN baseline tests passed successfully!")

