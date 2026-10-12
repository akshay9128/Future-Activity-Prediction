"""
CNN Feature Extractor Module.

Builds and configures a lightweight pretrained Convolutional Neural Network
(MobileNetV2) as a spatial feature extractor for video frames. Converts individual
frames into 1D visual embeddings (1280-dimensional vectors) and sequences of frames
into temporal feature matrices.
"""

from pathlib import Path
import sys
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def build_cnn_feature_extractor(
    backbone_name: str = "MobileNetV2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    trainable: bool = False,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """
    Construct a pretrained CNN feature extractor without classification top layers.

    Args:
        backbone_name: Name of architecture ('MobileNetV2').
        input_shape: (height, width, channels) of input frames.
        trainable: Whether CNN weights should be trainable or frozen.
        weights: Pretrained weights source ('imagenet' or None).

    Returns:
        tf.keras.Model: A model taking input (H, W, 3) and outputting a 1D vector (1280,).
    """
    if backbone_name.lower() == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling="avg",  # Global Average Pooling produces a 1D vector of length 1280
        )
    else:
        raise ValueError(f"Unsupported CNN backbone: '{backbone_name}'. Only 'MobileNetV2' supported.")

    base_model.trainable = trainable
    base_model._name = f"{backbone_name}_feature_extractor"
    return base_model


def build_sequence_feature_extractor(
    cnn_model: tf.keras.Model,
    sequence_length: int = 16,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """
    Wrap a frame-level CNN in a TimeDistributed layer to process entire video sequences.

    Args:
        cnn_model: Trained or pretrained frame-level CNN model.
        sequence_length: Number of time steps (frames) per sequence.
        input_shape: (H, W, C) shape of each individual frame.

    Returns:
        tf.keras.Model: Sequence model mapping (B, T, H, W, C) -> (B, T, feature_dim).
    """
    sequence_input = tf.keras.Input(
        shape=(sequence_length, *input_shape),
        dtype=tf.float32,
        name="video_sequence_input",
    )
    time_distributed_features = tf.keras.layers.TimeDistributed(
        cnn_model, name="time_distributed_cnn"
    )(sequence_input)

    return tf.keras.Model(
        inputs=sequence_input,
        outputs=time_distributed_features,
        name="video_sequence_feature_extractor",
    )


class CNNFeatureExtractor:
    """
    High-level helper class for extracting frame-level and sequence-level CNN embeddings.
    """

    def __init__(
        self,
        backbone_name: str = "MobileNetV2",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        trainable: bool = False,
        weights: str = "imagenet",
        sequence_length: int = 16,
    ):
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.frame_model = build_cnn_feature_extractor(
            backbone_name=backbone_name,
            input_shape=input_shape,
            trainable=trainable,
            weights=weights,
        )
        self.feature_dim = self.frame_model.output_shape[-1]
        self.sequence_model = build_sequence_feature_extractor(
            cnn_model=self.frame_model,
            sequence_length=sequence_length,
            input_shape=input_shape,
        )

    def extract_frame_features(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame (H, W, C) or a batch of frames (B, H, W, C).

        Returns:
            np.ndarray: (feature_dim,) if 3D input, or (B, feature_dim) if 4D input.
        """
        frames_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
        if len(frames.shape) == 3:
            # Single frame: add batch dimension
            frames_tensor = tf.expand_dims(frames_tensor, axis=0)
            features = self.frame_model(frames_tensor, training=False)
            return features.numpy().squeeze(axis=0)
        elif len(frames.shape) == 4:
            features = self.frame_model(frames_tensor, training=False)
            return features.numpy()
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {frames.shape}")

    def extract_sequence_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract features from a sequence (T, H, W, C) or batch of sequences (B, T, H, W, C).

        Returns:
            np.ndarray: (T, feature_dim) if 4D input, or (B, T, feature_dim) if 5D input.
        """
        seq_tensor = tf.convert_to_tensor(sequences, dtype=tf.float32)
        if len(sequences.shape) == 4:
            # Single video sequence: add batch dimension
            seq_tensor = tf.expand_dims(seq_tensor, axis=0)
            features = self.sequence_model(seq_tensor, training=False)
            return features.numpy().squeeze(axis=0)
        elif len(sequences.shape) == 5:
            features = self.sequence_model(seq_tensor, training=False)
            return features.numpy()
        else:
            raise ValueError(f"Expected 4D or 5D sequence tensor, got shape {sequences.shape}")


def main() -> None:
    """CLI demonstration of CNN feature extractor."""
    print("==================================================")
    print("      CNN Feature Extractor Verification (Phase 4)")
    print("==================================================")

    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
    config = load_config(config_path)

    backbone = config["model"]["backbone"]
    target_size = tuple(config["preprocessing"]["target_size"])
    seq_len = config["preprocessing"]["sequence_length"]

    print(f"Backbone Model   : {backbone}")
    print(f"Target Frame Size: {target_size}")
    print(f"Sequence Length  : {seq_len}")
    print("--------------------------------------------------")

    extractor = CNNFeatureExtractor(
        backbone_name=backbone,
        input_shape=(*target_size, 3),
        sequence_length=seq_len,
    )

    print(f"Feature Dimension: {extractor.feature_dim}")
    print(f"Frame Model Parameters: {extractor.frame_model.count_params():,}")
    print(f"Trainable Parameters  : {sum(v.numpy().size for v in extractor.frame_model.trainable_variables):,}")

    # Test 1: Single Frame
    single_frame = np.random.uniform(-1.0, 1.0, size=(*target_size, 3)).astype(np.float32)
    single_feat = extractor.extract_frame_features(single_frame)
    print(f"\n[Check 1] Single Frame Feature Shape: {single_feat.shape}")
    assert single_feat.shape == (1280,)
    assert np.all(np.isfinite(single_feat)), "Features contain NaN or Inf!"
    print("  -> Passed: Shape is (1280,) and all values are finite.")

    # Test 2: Batch of Frames
    batch_frames = np.random.uniform(-1.0, 1.0, size=(4, *target_size, 3)).astype(np.float32)
    batch_feat = extractor.extract_frame_features(batch_frames)
    print(f"\n[Check 2] Batch Frames Feature Shape: {batch_feat.shape}")
    assert batch_feat.shape == (4, 1280)
    assert np.all(np.isfinite(batch_feat)), "Batch features contain NaN or Inf!"
    print("  -> Passed: Shape is (4, 1280) and all values are finite.")

    # Test 3: Batch of Sequences
    batch_seq = np.random.uniform(-1.0, 1.0, size=(2, seq_len, *target_size, 3)).astype(np.float32)
    seq_feat = extractor.extract_sequence_features(batch_seq)
    print(f"\n[Check 3] Batch Sequences Feature Shape: {seq_feat.shape}")
    assert seq_feat.shape == (2, seq_len, 1280)
    assert np.all(np.isfinite(seq_feat)), "Sequence features contain NaN or Inf!"
    print("  -> Passed: Shape is (2, 16, 1280) and all values are finite.")

    print("\n[SUCCESS] CNN Feature Extractor verified successfully!")


if __name__ == "__main__":
    main()
