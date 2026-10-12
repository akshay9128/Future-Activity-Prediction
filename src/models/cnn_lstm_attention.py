"""
CNN + LSTM + Temporal Attention Architecture Module.

Implements the unified deep learning architecture combining frozen CNN spatial feature
extraction (MobileNetV2), sequential recurrent processing (LSTM with return_sequences=True),
and a custom temporal attention mechanism that computes frame-level importance weights.
"""

from pathlib import Path
import sys
from typing import Tuple
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_attention import TemporalAttention


def build_cnn_lstm_attention_head(
    sequence_length: int = 16,
    feature_dim: int = 1280,
    lstm_units: int = 64,
    attention_units: int = 64,
    dense_units: int = 64,
    num_classes: int = 5,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.0,
) -> tf.keras.Model:
    """
    Construct the recurrent temporal attention classification head.

    Flow:
        Inputs: (B, 16, 1280)
           │
           ▼
        LSTM(units=64, return_sequences=True) ──► (B, 16, 64)
           │
           ▼
        TemporalAttention(units=64) ────────────► Context: (B, 64), Weights: (B, 16, 1)
           │
           ▼
        Dense(64, activation='relu')
           │
           ▼
        Dropout(rate=0.3)
           │
           ▼
        Dense(5, activation='softmax') ─────────► (B, 5)

    Args:
        sequence_length: Number of time steps (16).
        feature_dim: Dimension of spatial feature vectors (1280).
        lstm_units: Hidden units for the recurrent LSTM layer.
        attention_units: Internal dimension for attention energy projection.
        dense_units: Units in classification projection layer.
        num_classes: Number of activity categories (5).
        dropout_rate: Dropout rate for regularization.
        recurrent_dropout: Dropout for LSTM recurrent connections (0.0 for cuDNN speed).

    Returns:
        tf.keras.Model: Classifier head taking sequential features to action probabilities.
    """
    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        dtype=tf.float32,
        name="sequential_features_input",
    )

    # 1. Recurrent LSTM over all time steps (must return full sequence of states)
    lstm_layer = tf.keras.layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        name="temporal_lstm",
    )
    lstm_sequence = lstm_layer(inputs)  # Shape: (B, 16, lstm_units)

    # 2. Temporal Attention Layer: pools 16 timesteps into a dynamic context vector
    attention_layer = TemporalAttention(
        units=attention_units,
        return_attention_scores=True,
        name="temporal_attention",
    )
    context_vector, _ = attention_layer(lstm_sequence)  # context: (B, lstm_units), weights: (B, 16, 1)

    # 3. Dense Projection and Regularization
    x = tf.keras.layers.Dense(
        dense_units, activation="relu", name="attention_dense_projection"
    )(context_vector)
    x = tf.keras.layers.Dropout(dropout_rate, name="attention_dropout")(x)

    # 4. Softmax Action Classification
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="action_probabilities"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_lstm_attention_classifier_head",
    )


def get_attention_extraction_model(head_model: tf.keras.Model) -> tf.keras.Model:
    """
    Build an auxiliary model sharing weights with the classification head that outputs
    both the action class probabilities and the temporal attention weights (B, T, 1).

    Args:
        head_model: The trained cnn_lstm_attention_classifier_head model.

    Returns:
        tf.keras.Model: Model outputting [probabilities, attention_weights].
    """
    attention_layer = head_model.get_layer("temporal_attention")
    attention_weights_tensor = attention_layer.output[1]

    return tf.keras.Model(
        inputs=head_model.input,
        outputs=[head_model.output, attention_weights_tensor],
        name="attention_extractor",
    )


def build_end_to_end_cnn_lstm_attention(
    cnn_frame_model: tf.keras.Model,
    attention_head_model: tf.keras.Model,
    sequence_length: int = 16,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """
    Construct complete end-to-end video classification model linking raw video frames
    to the CNN + LSTM + Attention classifier.

    Args:
        cnn_frame_model: Pretrained frozen CNN taking (224, 224, 3) -> (1280,).
        attention_head_model: Head taking (16, 1280) -> (num_classes,).
        sequence_length: Number of frames per video (16).
        input_shape: Resolution of each frame (H, W, C).

    Returns:
        tf.keras.Model: End-to-end model (B, T, H, W, C) -> (B, num_classes).
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

    predictions = attention_head_model(spatial_features)

    return tf.keras.Model(
        inputs=raw_video_input,
        outputs=predictions,
        name="cnn_lstm_attention_end_to_end",
    )
