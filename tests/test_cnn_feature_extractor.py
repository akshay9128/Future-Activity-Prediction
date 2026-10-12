"""
Phase 4 Test Suite: Pretrained CNN feature extractor (MobileNetV2).
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

from src.models.cnn_feature_extractor import (
    CNNFeatureExtractor,
    build_cnn_feature_extractor,
    build_sequence_feature_extractor,
)
from src.data.frame_sampler import load_and_sample_video
from src.data.dataset_manifest import load_manifest


@pytest.fixture(scope="module")
def feature_extractor():
    """Build a reusable CNN feature extractor instance for testing."""
    return CNNFeatureExtractor(
        backbone_name="MobileNetV2",
        input_shape=(224, 224, 3),
        trainable=False,
        sequence_length=16,
    )


def test_single_frame_feature_shape_and_finiteness(feature_extractor):
    """Verify that a single frame produces a finite 1280-dimensional feature vector."""
    dummy_frame = np.random.uniform(-1.0, 1.0, size=(224, 224, 3)).astype(np.float32)
    feat = feature_extractor.extract_frame_features(dummy_frame)

    assert feat.shape == (1280,), f"Expected shape (1280,), got {feat.shape}"
    assert np.all(np.isfinite(feat)), "Feature vector contains NaN or Inf values!"
    assert not np.all(feat == 0.0), "Feature vector is entirely zero!"


def test_batch_frames_processing(feature_extractor):
    """Verify that a batch of frames produces a (B, 1280) feature matrix."""
    batch_size = 4
    batch_frames = np.random.uniform(-1.0, 1.0, size=(batch_size, 224, 224, 3)).astype(np.float32)
    feats = feature_extractor.extract_frame_features(batch_frames)

    assert feats.shape == (batch_size, 1280), f"Expected shape ({batch_size}, 1280), got {feats.shape}"
    assert np.all(np.isfinite(feats)), "Batch features contain NaN or Inf values!"


def test_sequence_feature_extraction(feature_extractor):
    """Verify that video sequences produce (B, T, 1280) temporal features."""
    batch_size = 2
    seq_len = 16
    batch_seq = np.random.uniform(-1.0, 1.0, size=(batch_size, seq_len, 224, 224, 3)).astype(np.float32)
    seq_feats = feature_extractor.extract_sequence_features(batch_seq)

    assert seq_feats.shape == (batch_size, seq_len, 1280), (
        f"Expected shape ({batch_size}, {seq_len}, 1280), got {seq_feats.shape}"
    )
    assert np.all(np.isfinite(seq_feats)), "Sequence features contain NaN or Inf!"


def test_cnn_weights_are_frozen(feature_extractor):
    """Verify that CNN feature extractor is frozen (0 trainable variables)."""
    trainable_vars = feature_extractor.frame_model.trainable_variables
    assert len(trainable_vars) == 0, (
        f"Expected 0 trainable variables for frozen feature extractor, found {len(trainable_vars)}"
    )


def test_actual_video_frame_through_cnn(feature_extractor):
    """Pass frames from an actual UCF101 video through the CNN extractor."""
    manifest_path = PROJECT_ROOT / "outputs" / "dataset_manifest.csv"
    assert manifest_path.is_file(), "Dataset manifest not found!"

    df = load_manifest(manifest_path)
    sample_path = df.iloc[0]["video_path"]

    # Sample video sequence normalized in [-1.0, 1.0]
    seq = load_and_sample_video(
        sample_path,
        sequence_length=16,
        target_size=(224, 224),
        observation_ratio=0.5,
        normalization="mobilenet_v2",
    )

    # Extract features for all 16 frames
    seq_features = feature_extractor.extract_sequence_features(seq)

    assert seq_features.shape == (16, 1280)
    assert np.all(np.isfinite(seq_features))
    # Check that consecutive frames are not identical (there is motion/temporal variation)
    diff = np.abs(seq_features[0] - seq_features[-1]).sum()
    assert diff > 0.0, "Features across frames should reflect temporal visual changes"


if __name__ == "__main__":
    extractor = CNNFeatureExtractor(
        backbone_name="MobileNetV2",
        input_shape=(224, 224, 3),
        trainable=False,
        sequence_length=16,
    )
    test_single_frame_feature_shape_and_finiteness(extractor)
    test_batch_frames_processing(extractor)
    test_sequence_feature_extraction(extractor)
    test_cnn_weights_are_frozen(extractor)
    test_actual_video_frame_through_cnn(extractor)
    print("All Phase 4 CNN feature extractor tests passed successfully!")
