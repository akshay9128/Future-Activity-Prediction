"""
Test Suite for Temporal Attention Mechanism and CNN + LSTM + Attention Model.

Validates the custom TemporalAttention Keras layer, output dimensions, softmax weight
constraints, gradient backpropagation, layer serialization, and end-to-end inference.
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm_attention import (
    build_cnn_lstm_attention_head,
    build_end_to_end_cnn_lstm_attention,
    get_attention_extraction_model,
)
from src.models.temporal_attention import TemporalAttention


def test_temporal_attention_output_shapes():
    """Verify that TemporalAttention produces correct context and weight tensor shapes."""
    batch_size = 4
    seq_len = 16
    hidden_dim = 64
    attn_units = 32

    inputs = tf.random.normal((batch_size, seq_len, hidden_dim))

    # Without return_attention_scores
    attn_layer_single = TemporalAttention(units=attn_units, return_attention_scores=False)
    context_only = attn_layer_single(inputs)
    assert context_only.shape == (batch_size, hidden_dim), (
        f"Expected context shape ({batch_size}, {hidden_dim}), got {context_only.shape}"
    )

    # With return_attention_scores=True
    attn_layer_both = TemporalAttention(units=attn_units, return_attention_scores=True)
    context, weights = attn_layer_both(inputs)
    assert context.shape == (batch_size, hidden_dim), (
        f"Expected context shape ({batch_size}, {hidden_dim}), got {context.shape}"
    )
    assert weights.shape == (batch_size, seq_len, 1), (
        f"Expected weights shape ({batch_size}, {seq_len}, 1), got {weights.shape}"
    )


def test_temporal_attention_weights_sum_to_one():
    """Verify that temporal attention weights are non-negative and sum to 1.0 along the time axis."""
    batch_size = 8
    seq_len = 16
    hidden_dim = 64

    inputs = tf.random.normal((batch_size, seq_len, hidden_dim))
    attn_layer = TemporalAttention(units=32, return_attention_scores=True)
    _, weights = attn_layer(inputs)

    weights_np = weights.numpy()

    # All weights must be non-negative
    assert np.all(weights_np >= 0.0), "Attention weights must be non-negative probabilities"

    # Sum along sequence dimension (axis 1) should equal 1.0 for every sample
    weight_sums = np.sum(weights_np, axis=1)
    np.testing.assert_allclose(
        weight_sums,
        np.ones((batch_size, 1)),
        atol=1e-5,
        err_msg="Attention weights must sum to 1.0 along the temporal sequence dimension",
    )


def test_gradient_flow_through_attention():
    """Verify non-zero, finite gradients flow through attention weights and back to inputs."""
    batch_size = 2
    seq_len = 16
    hidden_dim = 32

    inputs = tf.Variable(tf.random.normal((batch_size, seq_len, hidden_dim)), trainable=True)
    attn_layer = TemporalAttention(units=16, return_attention_scores=False)

    with tf.GradientTape(persistent=True) as tape:
        context = attn_layer(inputs)
        loss = tf.reduce_mean(tf.square(context))

    # Gradient with respect to inputs
    grad_inputs = tape.gradient(loss, inputs)
    assert grad_inputs is not None, "Gradient with respect to inputs must not be None"
    assert not np.isnan(grad_inputs.numpy()).any(), "Gradient wrt inputs contains NaNs"
    assert np.any(grad_inputs.numpy() != 0.0), "Gradient wrt inputs is all zeros"

    # Gradients with respect to trainable layer weights
    for weight in attn_layer.trainable_variables:
        grad_w = tape.gradient(loss, weight)
        assert grad_w is not None, f"Gradient for {weight.name} must not be None"
        assert not np.isnan(grad_w.numpy()).any(), f"Gradient for {weight.name} contains NaNs"
        assert np.any(grad_w.numpy() != 0.0), f"Gradient for {weight.name} is all zeros"


def test_attention_layer_serialization():
    """Verify that TemporalAttention can be saved and deserialized via get_config / from_config."""
    original_layer = TemporalAttention(units=48, return_attention_scores=True, name="custom_attn")
    config = original_layer.get_config()

    assert config["units"] == 48
    assert config["return_attention_scores"] is True

    recreated_layer = TemporalAttention.from_config(config)
    assert recreated_layer.units == 48
    assert recreated_layer.return_attention_scores is True


def test_cnn_lstm_attention_head_training_step():
    """Verify that a forward/backward training step on the classifier head reduces loss."""
    head_model = build_cnn_lstm_attention_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=32,
        attention_units=32,
        dense_units=32,
        num_classes=5,
        dropout_rate=0.2,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    dummy_x = tf.random.normal((8, 16, 1280))
    dummy_y = tf.constant([0, 1, 2, 3, 4, 0, 1, 2], dtype=tf.int32)

    with tf.GradientTape() as tape:
        preds = head_model(dummy_x, training=True)
        initial_loss = loss_fn(dummy_y, preds)

    grads = tape.gradient(initial_loss, head_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, head_model.trainable_variables))

    preds_after = head_model(dummy_x, training=False)
    loss_after = loss_fn(dummy_y, preds_after)

    assert not np.isnan(initial_loss.numpy()), "Initial loss is NaN"
    assert not np.isnan(loss_after.numpy()), "Loss after step is NaN"
    assert loss_after.numpy() < initial_loss.numpy(), "Training step failed to decrease loss"


def test_attention_extraction_model():
    """Verify that get_attention_extraction_model extracts valid attention probabilities and weights."""
    head_model = build_cnn_lstm_attention_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=32,
        attention_units=32,
        num_classes=5,
    )
    extractor = get_attention_extraction_model(head_model)

    dummy_x = tf.random.normal((3, 16, 1280))
    probs, weights = extractor(dummy_x)

    assert probs.shape == (3, 5), f"Expected probabilities shape (3, 5), got {probs.shape}"
    assert weights.shape == (3, 16, 1), f"Expected weights shape (3, 16, 1), got {weights.shape}"

    # Verify probability distribution properties
    np.testing.assert_allclose(
        np.sum(probs.numpy(), axis=1),
        np.ones(3),
        atol=1e-5,
        err_msg="Action probabilities must sum to 1.0",
    )
    np.testing.assert_allclose(
        np.sum(weights.numpy(), axis=1),
        np.ones((3, 1)),
        atol=1e-5,
        err_msg="Attention weights must sum to 1.0",
    )


def test_cnn_lstm_attention_head_output_shape_101():
    """Verify that the CNN + LSTM + Attention head maps (B, 16, 1280) to (B, 101)."""
    model = build_cnn_lstm_attention_head(
        sequence_length=16,
        feature_dim=1280,
        lstm_units=32,
        attention_units=32,
        num_classes=101,
    )
    dummy_input = tf.random.normal((4, 16, 1280))
    output = model(dummy_input, training=False)

    assert output.shape == (4, 101), f"Expected shape (4, 101), got {output.shape}"
    prob_sums = np.sum(output.numpy(), axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
    assert np.all(output.numpy() >= 0.0), "Probabilities must be non-negative"


def test_saved_artifacts_exist():
    """Verify that training artifacts, models, and visualization plots exist on disk."""
    artifacts = [
        Path("outputs/models/cnn_lstm_attention_head.keras"),
        Path("outputs/models/cnn_lstm_attention_end_to_end.keras"),
        Path("outputs/logs/cnn_lstm_attention_training.png"),
        Path("outputs/logs/cnn_lstm_attention_confusion_matrix.png"),
        Path("outputs/logs/cnn_lstm_attention_val_metrics.json"),
        Path("outputs/logs/temporal_attention_weights_visualization.png"),
    ]
    for artifact in artifacts:
        assert artifact.exists(), f"Expected artifact {artifact} does not exist"


def test_end_to_end_cnn_lstm_attention_prediction():
    """Verify end-to-end 5D raw video tensor inference (B, T, H, W, C) -> (B, num_classes)."""
    e2e_model_path = Path("outputs/models/cnn_lstm_attention_end_to_end.keras")
    assert e2e_model_path.exists(), "End-to-end model artifact missing"

    e2e_model = tf.keras.models.load_model(e2e_model_path)

    # Single dummy video with 16 frames of resolution 224x224x3
    dummy_video = tf.random.uniform((1, 16, 224, 224, 3), minval=-1.0, maxval=1.0)
    preds = e2e_model(dummy_video, training=False)

    assert preds.shape in ((1, 5), (1, 101)), f"Expected end-to-end prediction shape (1, 5) or (1, 101), got {preds.shape}"
    assert np.isclose(np.sum(preds.numpy()), 1.0, atol=1e-4), "Predictions must sum to 1.0"
