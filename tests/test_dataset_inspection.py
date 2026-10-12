"""
Phase 2 Test Suite: UCF101 dataset verification and class inspection.
"""

from pathlib import Path
import sys
import tempfile
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_inspector import (
    inspect_dataset,
    inspect_video_file,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from src.utils.config import load_config



@pytest.fixture(scope="module")
def project_config():
    """Load default project configuration."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
    return load_config(config_path)


def test_dataset_root_and_classes_exist(project_config):
    """Verify that UCF101 dataset directory exists and all 5 requested classes are present."""
    raw_dir = Path(project_config["dataset"]["raw_dir"])
    selected_classes = project_config["dataset"]["selected_classes"]

    assert raw_dir.is_dir(), f"Dataset directory not found at: {raw_dir}"

    for class_name in selected_classes:
        class_folder = raw_dir / class_name
        assert class_folder.is_dir(), f"Class folder missing: {class_folder}"


def test_video_counts_per_class(project_config):
    """Verify each selected class folder contains a reasonable number of videos (>50)."""
    raw_dir = Path(project_config["dataset"]["raw_dir"])
    selected_classes = project_config["dataset"]["selected_classes"]

    for class_name in selected_classes:
        class_folder = raw_dir / class_name
        videos = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        assert len(videos) > 50, (
            f"Class '{class_name}' has insufficient videos: {len(videos)} found."
        )


def test_sample_videos_readable(project_config):
    """Verify that OpenCV can successfully open and read frames from each class."""
    raw_dir = Path(project_config["dataset"]["raw_dir"])
    selected_classes = project_config["dataset"]["selected_classes"]

    for class_name in selected_classes:
        class_folder = raw_dir / class_name
        video_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        assert len(video_files) > 0, f"No videos found in {class_folder}"

        sample_video = sorted(video_files)[0]
        meta = inspect_video_file(sample_video)

        assert meta["valid"] is True, f"Failed to read sample video: {sample_video}, error: {meta['error']}"
        assert meta["frame_count"] > 0, f"Video has 0 frames: {sample_video}"
        assert meta["fps"] > 0, f"Video has non-positive FPS: {sample_video}"
        assert meta["width"] > 0 and meta["height"] > 0, f"Invalid dimensions: {sample_video}"
        assert meta["error"] is None, f"Unexpected error reported: {meta['error']}"


def test_unreadable_and_missing_videos_reported():
    """Verify that missing and corrupted files report errors rather than failing silently."""
    # Test 1: Non-existent file
    missing_path = Path("D:/datasets/non_existent_video_12345.avi")
    res_missing = inspect_video_file(missing_path)
    assert res_missing["valid"] is False
    assert "not found" in res_missing["error"].lower()

    # Test 2: Empty / 0-byte corrupted file
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        res_empty = inspect_video_file(tmp_path)
        assert res_empty["valid"] is False
        assert "empty" in res_empty["error"].lower() or "0 bytes" in res_empty["error"].lower()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_dataset_summary_json_file_created():
    """Verify that outputs/dataset_summary.json exists and is valid JSON."""
    summary_path = Path(__file__).resolve().parent.parent / "outputs" / "dataset_summary.json"
    assert summary_path.is_file(), f"Summary file missing: {summary_path}"
    assert summary_path.stat().st_size > 0, "Summary file is empty"


if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml")
    test_dataset_root_and_classes_exist(cfg)
    test_video_counts_per_class(cfg)
    test_sample_videos_readable(cfg)
    test_unreadable_and_missing_videos_reported()
    test_dataset_summary_json_file_created()
    print("All Phase 2 dataset inspection tests passed successfully!")
