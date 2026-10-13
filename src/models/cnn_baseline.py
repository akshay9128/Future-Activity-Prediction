"""
CNN-Only Baseline Architecture Module.

Implements the non-temporal CNN baseline model. Processes video frame sequences
by performing Global Temporal Average Pooling over frame-level CNN features,
followed by dense classification layers.
"""

from pathlib import Path
import sys
from typing import Tuple
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_cnn_baseline_head(
    sequence_length: int = 16,
    feature_dim: int = 1280,
    dense_units: int = 64,
    num_classes: int = 101,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build the classification head taking (sequence_length, feature_dim) and predicting classes.

    Args:
        sequence_length: Temporal length (number of frames).
        feature_dim: CNN embedding dimension (1280 for MobileNetV2).
        dense_units: Number of neurons in intermediate dense layer.
        num_classes: Number of target action categories (default 101 for full UCF-101).
        dropout_rate: Dropout rate for regularization.

    Returns:
        tf.keras.Model: Keras model mapping (B, T, 1280) -> (B, num_classes).
    """
    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        dtype=tf.float32,
        name="temporal_features_input",
    )
    # Global Temporal Average Pooling: pools over time dimension T -> (B, feature_dim)
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="temporal_avg_pool")(inputs)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_projection")(pooled)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_regularization")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="action_probabilities"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_only_baseline_head",
    )


def build_end_to_end_cnn_baseline(
    cnn_frame_model: tf.keras.Model,
    head_model: tf.keras.Model,
    sequence_length: int = 16,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """
    Connect the raw video input through the frozen frame CNN to the classification head.

    Args:
        cnn_frame_model: Frozen MobileNetV2 model taking (224, 224, 3) -> (1280,).
        head_model: Baseline classifier taking (T, 1280) -> (num_classes,).
        sequence_length: Number of frames per video.
        input_shape: Frame dimensions (H, W, C).

    Returns:
        tf.keras.Model: End-to-end model mapping (B, T, H, W, C) -> (B, num_classes).
    """
    video_input = tf.keras.Input(
        shape=(sequence_length, *input_shape),
        dtype=tf.float32,
        name="video_frames_input",
    )
    # Apply frozen CNN across all frames
    features = tf.keras.layers.TimeDistributed(
        cnn_frame_model, name="timedistributed_cnn"
    )(video_input)
    predictions = head_model(features)

    return tf.keras.Model(
        inputs=video_input,
        outputs=predictions,
        name="cnn_only_baseline_end_to_end",
    )
