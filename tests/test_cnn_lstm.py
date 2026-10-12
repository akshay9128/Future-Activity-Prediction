"""
Test Suite: CNN + LSTM architecture, training step, and gradient verification.
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

from src.models.cnn_lstm import (
    build_cnn_lstm_head,
    build_end_to_end_cnn_lstm,
)


def test_cnn_lstm_head_output_shape():
    """Verify that the CNN + LSTM head maps (B, 16, 1280) to (B, 5)."""
    model = build_cnn_lstm_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=64,
        num_classes=5,
    )
    dummy_input = np.random.uniform(-1.0, 1.0, size=(4, 16, 1280)).astype(np.float32)
    output = model(dummy_input, training=False)

    assert output.shape == (4, 5), f"Expected shape (4, 5), got {output.shape}"
    prob_sums = np.sum(output.numpy(), axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
    assert np.all(output.numpy() >= 0.0), "Class probabilities must be non-negative"


def test_cnn_lstm_sequence_representation_shape():
    """Verify return_sequences=True yields (B, 16, lstm_units) for attention integration."""
    seq_model = build_cnn_lstm_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=64,
        return_sequences=True,
    )
    dummy_input = np.random.uniform(-1.0, 1.0, size=(2, 16, 1280)).astype(np.float32)
    seq_out = seq_model(dummy_input, training=False)

    assert seq_out.shape == (2, 16, 64)
    assert np.all(np.isfinite(seq_out.numpy()))


def test_gradient_flow_and_finiteness():
    """Verify that backpropagation computes finite gradients for all LSTM trainable variables."""
    model = build_cnn_lstm_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=32,
        num_classes=5,
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    x = tf.random.normal((2, 16, 1280))
    y = tf.constant([1, 3], dtype=tf.int64)

    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y, preds)

    grads = tape.gradient(loss, model.trainable_variables)

    assert len(grads) == len(model.trainable_variables)
    for g in grads:
        assert g is not None, "Gradient should not be None"
        assert np.all(np.isfinite(g.numpy())), "Gradients contain NaN or Inf values!"


def test_cnn_lstm_training_step():
    """Verify model trains for 2 epochs on dummy data with finite loss."""
    model = build_cnn_lstm_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=32,
        num_classes=5,
    )
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
        assert np.isfinite(loss_val) and loss_val > 0.0


def test_saved_cnn_lstm_model_reload():
    """Verify saved CNN + LSTM model reloads and reproduces exact predictions."""
    head_path = PROJECT_ROOT / "outputs" / "models" / "cnn_lstm_head.keras"
    assert head_path.is_file(), f"Model file missing: {head_path}"

    reloaded = tf.keras.models.load_model(head_path)
    dummy_input = np.ones((2, 16, 1280), dtype=np.float32)

    pred1 = reloaded.predict(dummy_input, verbose=0)
    pred2 = reloaded.predict(dummy_input, verbose=0)

    np.testing.assert_allclose(pred1, pred2, atol=1e-6)
    assert pred1.shape == (2, 5)


def test_cnn_lstm_artifacts_exist():
    """Verify all CNN + LSTM output models, plots, and logs exist."""
    expected = [
        PROJECT_ROOT / "outputs" / "models" / "cnn_lstm_head.keras",
        PROJECT_ROOT / "outputs" / "models" / "cnn_lstm_end_to_end.keras",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_lstm_training.png",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_lstm_confusion_matrix.png",
        PROJECT_ROOT / "outputs" / "logs" / "cnn_lstm_val_metrics.json",
    ]
    for path in expected:
        assert path.is_file(), f"Expected artifact missing: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"


def test_end_to_end_cnn_lstm_prediction():
    """Verify end-to-end model accepts raw 5D video frames (1, 16, 224, 224, 3)."""
    e2e_path = PROJECT_ROOT / "outputs" / "models" / "cnn_lstm_end_to_end.keras"
    assert e2e_path.is_file(), f"End-to-end model missing: {e2e_path}"

    e2e_model = tf.keras.models.load_model(e2e_path)
    raw_video = np.random.uniform(-1.0, 1.0, size=(1, 16, 224, 224, 3)).astype(np.float32)
    probs = e2e_model.predict(raw_video, verbose=0)

    assert probs.shape == (1, 5)
    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-4)


if __name__ == "__main__":
    test_cnn_lstm_head_output_shape()
    test_cnn_lstm_sequence_representation_shape()
    test_gradient_flow_and_finiteness()
    test_cnn_lstm_training_step()
    test_saved_cnn_lstm_model_reload()
    test_cnn_lstm_artifacts_exist()
    test_end_to_end_cnn_lstm_prediction()
    print("All CNN + LSTM tests passed successfully!")
