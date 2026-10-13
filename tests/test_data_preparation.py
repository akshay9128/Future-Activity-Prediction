"""
Test Suite: Data preparation, temporal frame sampling, and leakage-safe splits.
"""


from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frame_sampler import (
    compute_sample_indices,
    load_and_sample_video,
    normalize_frames,
    VideoLoadingError,
)
from src.data.dataset_manifest import (
    build_dataset_manifest,
    load_manifest,
    extract_group_id,
)
from src.data.feature_cache import validate_cache_archive
from src.data.video_dataset import VideoSequenceDataset
from src.utils.config import load_config


def _get_manifest_df():
    """Load or generate the dataset manifest for testing."""
    manifest_path = PROJECT_ROOT / "outputs" / "dataset_manifest.csv"
    if not manifest_path.is_file():
        cfg = load_config(PROJECT_ROOT / "configs" / "default_config.yaml")
        df = build_dataset_manifest(
            dataset_dir=cfg["dataset"]["raw_dir"],
            selected_classes=cfg["dataset"]["selected_classes"],
            train_split=cfg["dataset"]["train_split"],
            val_split=cfg["dataset"]["val_split"],
            test_split=cfg["dataset"]["test_split"],
            seed=cfg.get("seed", 42),
        )
        return df
    return load_manifest(manifest_path)


@pytest.fixture(scope="module")
def manifest_df():
    """Pytest fixture wrapping _get_manifest_df."""
    return _get_manifest_df()



def test_compute_sample_indices_early_portion():
    """Verify that indices are uniformly sampled only within the early observation portion."""
    total_frames = 100
    seq_len = 16
    obs_ratio = 0.5  # observe frames 0 to 49

    indices = compute_sample_indices(
        total_frames=total_frames,
        sequence_length=seq_len,
        observation_ratio=obs_ratio,
    )
    assert len(indices) == seq_len
    assert indices[0] == 0
    assert indices[-1] < int(total_frames * obs_ratio)
    assert np.all(np.diff(indices) >= 0), "Sample indices must be monotonically non-decreasing"


def test_frame_normalization_ranges():
    """Verify that normalization maps pixel values to expected numeric intervals."""
    dummy_pixels = np.array([0, 127.5, 255], dtype=np.float32).reshape(1, 1, 1, 3)

    norm_mobilenet = normalize_frames(dummy_pixels, mode="mobilenet_v2")
    assert np.isclose(norm_mobilenet.min(), -1.0)
    assert np.isclose(norm_mobilenet.max(), 1.0)

    norm_zero_one = normalize_frames(dummy_pixels, mode="zero_one")
    assert np.isclose(norm_zero_one.min(), 0.0)
    assert np.isclose(norm_zero_one.max(), 1.0)


def test_load_and_sample_video_output(manifest_df):
    """Verify shape, dtype, and value bounds on an actual video from the manifest."""
    sample_video_path = manifest_df.iloc[0]["video_path"]
    seq = load_and_sample_video(
        video_path=sample_video_path,
        sequence_length=16,
        target_size=(224, 224),
        observation_ratio=0.5,
        normalization="mobilenet_v2",
    )
    assert seq.shape == (16, 224, 224, 3)
    assert seq.dtype == np.float32
    assert -1.0 <= seq.min() <= 1.0
    assert -1.0 <= seq.max() <= 1.0


def test_short_video_handling_without_crashing():
    """Verify that videos with fewer frames than sequence_length are handled gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Create a small synthetic video with only 5 frames
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(tmp_path), fourcc, 10.0, (64, 64))
        for _ in range(5):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()

        # Request 16 frames from a 5-frame video
        seq = load_and_sample_video(
            video_path=tmp_path,
            sequence_length=16,
            target_size=(64, 64),
            observation_ratio=0.5,
            normalization="mobilenet_v2",
        )
        assert seq.shape == (16, 64, 64, 3)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_unreadable_video_raises_video_loading_error():
    """Verify that non-existent or corrupted files raise VideoLoadingError."""
    with pytest.raises(VideoLoadingError):
        load_and_sample_video("D:/non_existent_path_test.avi")


def test_manifest_disjoint_splits(manifest_df):
    """CRITICAL: Confirm zero data leakage between train, val, and test splits."""
    train_paths = set(manifest_df[manifest_df["split"] == "train"]["video_path"])
    val_paths = set(manifest_df[manifest_df["split"] == "val"]["video_path"])
    test_paths = set(manifest_df[manifest_df["split"] == "test"]["video_path"])

    assert len(train_paths & val_paths) == 0, "Train and Val share video paths (Data Leakage!)"
    assert len(train_paths & test_paths) == 0, "Train and Test share video paths (Data Leakage!)"
    assert len(val_paths & test_paths) == 0, "Val and Test share video paths (Data Leakage!)"


def test_split_reproducibility():
    """Verify that repeated runs with identical seed produce the exact same split."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default_config.yaml")
    df1 = build_dataset_manifest(
        dataset_dir=cfg["dataset"]["raw_dir"],
        selected_classes=cfg["dataset"]["selected_classes"],
        seed=42,
    )
    df2 = build_dataset_manifest(
        dataset_dir=cfg["dataset"]["raw_dir"],
        selected_classes=cfg["dataset"]["selected_classes"],
        seed=42,
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_video_dataset_batch_generation(manifest_df):
    """Verify that VideoSequenceDataset produces correct batch tensors and labels."""
    dataset = VideoSequenceDataset(
        manifest_df=manifest_df,
        split="train",
        sequence_length=8,
        target_size=(112, 112),
        observation_ratio=0.5,
        normalization="mobilenet_v2",
    )
    batch_x, batch_y = dataset.get_batch(batch_size=4)

    assert batch_x.shape == (4, 8, 112, 112, 3)
    assert batch_x.dtype == np.float32
    assert batch_y.shape == (4,)
    num_classes = manifest_df["class_idx"].nunique()
    assert np.all((batch_y >= 0) & (batch_y < num_classes))


def test_class_to_idx_mapping_101():
    """Verify class_to_idx.json and idx_to_class.json contain all 101 classes."""
    import json

    c2i_path = PROJECT_ROOT / "outputs" / "class_to_idx.json"
    i2c_path = PROJECT_ROOT / "outputs" / "idx_to_class.json"

    assert c2i_path.is_file(), f"Missing mapping file: {c2i_path}"
    assert i2c_path.is_file(), f"Missing reverse mapping file: {i2c_path}"

    with open(c2i_path, "r", encoding="utf-8") as f:
        c2i = json.load(f)
    with open(i2c_path, "r", encoding="utf-8") as f:
        i2c = json.load(f)

    assert len(c2i) == 101, f"Expected 101 classes in mapping, got {len(c2i)}"
    assert len(i2c) == 101, f"Expected 101 classes in reverse mapping, got {len(i2c)}"
    assert min(c2i.values()) == 0
    assert max(c2i.values()) == 100


def test_all_101_classes_represented_in_all_splits(manifest_df):
    """Verify that all 101 classes are represented in train, val, and test splits."""
    assert len(manifest_df) == 13320, f"Expected 13320 total videos, found {len(manifest_df)}"
    assert manifest_df["class_name"].nunique() == 101

    train_df = manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "val"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    assert len(train_df) == 9324
    assert len(val_df) == 1998
    assert len(test_df) == 1998

    assert train_df["class_name"].nunique() == 101, "Not all 101 classes in train split"
    assert val_df["class_name"].nunique() == 101, "Not all 101 classes in val split"
    assert test_df["class_name"].nunique() == 101, "Not all 101 classes in test split"

    # Verify every class has minimum sample representation
    train_counts = train_df["class_name"].value_counts()
    val_counts = val_df["class_name"].value_counts()
    test_counts = test_df["class_name"].value_counts()

    assert (train_counts >= 70).all(), f"Some classes have fewer than 70 train videos: {train_counts.min()}"
    assert (val_counts >= 15).all(), f"Some classes have fewer than 15 val videos: {val_counts.min()}"
    assert (test_counts >= 15).all(), f"Some classes have fewer than 15 test videos: {test_counts.min()}"


def test_feature_cache_archive_validation(tmp_path):
    """Verify validate_cache_archive detects valid and corrupted cache archives."""
    valid_cache = tmp_path / "valid_test.npz"
    features = np.random.uniform(-1.0, 1.0, size=(10, 16, 1280)).astype(np.float32)
    labels = np.arange(10, dtype=np.int64)
    np.savez_compressed(valid_cache, features=features, labels=labels)

    check_valid = validate_cache_archive(
        cache_path=valid_cache,
        expected_samples=10,
        expected_seq_len=16,
        expected_feat_dim=1280,
    )
    assert check_valid["valid"] is True
    assert check_valid["samples"] == 10
    assert check_valid["sequence_length"] == 16
    assert check_valid["feature_dim"] == 1280

    # Mismatch sample count
    check_mismatch = validate_cache_archive(
        cache_path=valid_cache,
        expected_samples=99,
    )
    assert check_mismatch["valid"] is False
    assert "Sample count mismatch" in check_mismatch["error"]

    # Non-existent archive
    check_missing = validate_cache_archive(tmp_path / "missing.npz")
    assert check_missing["valid"] is False
    assert "does not exist" in check_missing["error"]


def test_configurable_sequence_and_size_parameters(manifest_df):
    """Verify load_and_sample_video works across different sequence lengths and resolutions."""
    sample_vid = manifest_df.iloc[0]["video_path"]

    # Test short sequence length 8, smaller resolution (112, 112)
    seq_8 = load_and_sample_video(
        video_path=sample_vid,
        sequence_length=8,
        target_size=(112, 112),
        observation_ratio=0.3,
        normalization="mobilenet_v2",
    )
    assert seq_8.shape == (8, 112, 112, 3)
    assert seq_8.dtype == np.float32
    assert -1.0 <= seq_8.min() <= 1.0

    # Test sequence length 16, standard resolution (224, 224)
    seq_16 = load_and_sample_video(
        video_path=sample_vid,
        sequence_length=16,
        target_size=(224, 224),
        observation_ratio=0.5,
        normalization="mobilenet_v2",
    )
    assert seq_16.shape == (16, 224, 224, 3)


if __name__ == "__main__":
    df = _get_manifest_df()
    test_compute_sample_indices_early_portion()
    test_frame_normalization_ranges()
    test_load_and_sample_video_output(df)
    test_short_video_handling_without_crashing()
    test_unreadable_video_raises_video_loading_error()
    test_manifest_disjoint_splits(df)
    test_split_reproducibility()
    test_video_dataset_batch_generation(df)
    test_all_101_classes_represented_in_all_splits(df)
    test_configurable_sequence_and_size_parameters(df)
    print("All data preparation tests passed successfully!")


