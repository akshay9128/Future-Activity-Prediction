"""
CNN + LSTM Architecture Module.

Implements the temporal recurrent architecture combining frame-level CNN spatial
features with a Long Short-Term Memory (LSTM) network to model frame order, motion,
and temporal transitions.
"""

from pathlib import Path
import sys
from typing import Tuple
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_cnn_lstm_head(
    sequence_length: int = 16,
    feature_dim: int = 1280,
    lstm_units: int = 64,
    dense_units: int = 64,
    num_classes: int = 5,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.0,
    return_sequences: bool = False,
) -> tf.keras.Model:
    """
    Construct the recurrent LSTM classification head for sequential feature inputs.

    Args:
        sequence_length: Number of time steps (frames) per video sequence.
        feature_dim: Dimension of input CNN features at each time step (1280).
        lstm_units: Hidden dimensionality of the LSTM cell.
        dense_units: Number of neurons in the intermediate projection layer.
        num_classes: Number of output action categories (5).
        dropout_rate: Dropout rate for regularization.
        recurrent_dropout: Dropout applied to recurrent state transitions.
        return_sequences: If True, returns (B, T, lstm_units). If False, returns (B, num_classes).

    Returns:
        tf.keras.Model: Keras model processing sequential features into action predictions.
    """
    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        dtype=tf.float32,
        name="sequential_features_input",
    )

    # LSTM processes the 16 ordered frame embeddings sequentially
    lstm_layer = tf.keras.layers.LSTM(
        units=lstm_units,
        return_sequences=return_sequences,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        name="temporal_lstm",
    )
    lstm_out = lstm_layer(inputs)

    if return_sequences:
        # Returns hidden representations at each time step
        return tf.keras.Model(
            inputs=inputs,
            outputs=lstm_out,
            name="cnn_lstm_sequence_representation",
        )

    # Classification layers after temporal recurrence
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="lstm_dense_projection")(lstm_out)
    x = tf.keras.layers.Dropout(dropout_rate, name="lstm_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="action_probabilities"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_lstm_classifier_head",
    )


def build_end_to_end_cnn_lstm(
    cnn_frame_model: tf.keras.Model,
    lstm_head_model: tf.keras.Model,
    sequence_length: int = 16,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """
    Construct complete end-to-end video model linking raw frame inputs to the CNN + LSTM classifier.

    Args:
        cnn_frame_model: Pretrained frozen CNN taking (224, 224, 3) -> (1280,).
        lstm_head_model: Recurrent head taking (16, 1280) -> (num_classes,).
        sequence_length: Number of frames per video.
        input_shape: Resolution of each frame (H, W, C).

    Returns:
        tf.keras.Model: End-to-end video classification model (B, T, H, W, C) -> (B, num_classes).
    """
    raw_video_input = tf.keras.Input(
        shape=(sequence_length, *input_shape),
        dtype=tf.float32,
        name="raw_video_frames_input",
    )
    # Apply frozen CNN across all frames
    spatial_features = tf.keras.layers.TimeDistributed(
        cnn_frame_model, name="timedistributed_mobilenet"
    )(raw_video_input)
    predictions = lstm_head_model(spatial_features)

    return tf.keras.Model(
        inputs=raw_video_input,
        outputs=predictions,
        name="cnn_lstm_end_to_end",
    )
