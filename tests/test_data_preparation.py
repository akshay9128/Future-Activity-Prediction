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
    assert np.all((batch_y >= 0) & (batch_y < 5))


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
    print("All data preparation tests passed successfully!")


