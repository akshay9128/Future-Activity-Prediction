"""
Test Suite: End-to-end smoke test and reproducibility verification.
Ensures deterministic predictions, weight finiteness, and pipeline stability for the single CNN + LSTM model.
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from smoke_test import run_smoke_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_smoke_pipeline_passes():
    """Verify that the end-to-end smoke test passes without error."""
    success = run_smoke_test()
    assert success is True, "Pipeline smoke test failed"


def test_deterministic_reproducibility():
    """Verify that given identical inputs, the CNN + LSTM head produces identical predictions."""
    head_path = PROJECT_ROOT / "outputs" / "checkpoints" / "cnn_lstm_head.keras"
    model = tf.keras.models.load_model(head_path, compile=False)

    np.random.seed(42)
    sample_input = np.random.randn(2, 16, 1280).astype(np.float32)

    pred1 = model.predict(sample_input, verbose=0)
    pred2 = model.predict(sample_input, verbose=0)

    np.testing.assert_allclose(pred1, pred2, rtol=1e-6, atol=1e-6)
    assert pred1.shape == (2, 101)


def test_model_weight_finiteness():
    """Verify that all weights in the trained model checkpoint are finite numbers."""
    head_path = PROJECT_ROOT / "outputs" / "checkpoints" / "cnn_lstm_head.keras"
    model = tf.keras.models.load_model(head_path, compile=False)

    for weight_tensor in model.weights:
        arr = weight_tensor.numpy()
        assert np.isfinite(arr).all(), f"Found non-finite values in weight {weight_tensor.name}"
