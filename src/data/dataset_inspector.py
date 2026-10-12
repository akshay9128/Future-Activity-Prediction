"""
Dataset Inspector Module.

Inspects the UCF101 dataset on disk, verifies the presence of target action classes,
counts available videos, checks video readability using OpenCV, and generates a
summary report.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import cv2

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



SUPPORTED_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def inspect_video_file(video_path: Path | str) -> Dict[str, Any]:
    """
    Open a video file with OpenCV and inspect its properties and readability.

    Args:
        video_path: Path to the video file.

    Returns:
        dict: Video inspection status and metadata:
              - 'valid': bool (True if file opens and at least one frame is readable)
              - 'path': str
              - 'filename': str
              - 'frame_count': int
              - 'fps': float
              - 'width': int
              - 'height': int
              - 'error': Optional[str]
    """
    path = Path(video_path)
    result: Dict[str, Any] = {
        "valid": False,
        "path": str(path.resolve()),
        "filename": path.name,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "error": None,
    }

    if not path.is_file():
        result["error"] = f"File not found: {path}"
        return result

    if path.stat().st_size == 0:
        result["error"] = "File is empty (0 bytes)"
        return result

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result["error"] = "OpenCV failed to open video file (unsupported codec or corrupted header)"
            return result

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        success, frame = cap.read()
        if not success or frame is None:
            result["error"] = "Failed to read initial frame from video"
            return result

        result["valid"] = True
        result["frame_count"] = total_frames
        result["fps"] = round(fps, 2)
        result["width"] = width
        result["height"] = height
        return result
    finally:
        cap.release()


def inspect_dataset(
    dataset_dir: Path | str,
    selected_classes: List[str],
    samples_per_class: int = 3,
) -> Dict[str, Any]:
    """
    Inspect the UCF101 dataset directory for selected action classes.

    Args:
        dataset_dir: Root directory of UCF101 dataset.
        selected_classes: List of class folder names to verify.
        samples_per_class: Number of sample videos per class to test for readability.

    Returns:
        dict: Inspection summary containing class counts, sample checks, and status.
    """
    root = Path(dataset_dir)
    summary: Dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "root_exists": root.is_dir(),
        "total_selected_classes": len(selected_classes),
        "available_classes": [],
        "missing_classes": [],
        "class_video_counts": {},
        "total_videos_found": 0,
        "sample_inspections": {},
        "all_classes_valid": False,
    }

    if not root.is_dir():
        summary["error"] = f"Dataset root directory does not exist: {root}"
        return summary

    for class_name in selected_classes:
        class_folder = root / class_name
        if not class_folder.is_dir():
            summary["missing_classes"].append(class_name)
            continue

        summary["available_classes"].append(class_name)

        # Collect video files with supported extensions
        video_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        # Sort for deterministic ordering
        video_files.sort(key=lambda p: p.name)

        count = len(video_files)
        summary["class_video_counts"][class_name] = count
        summary["total_videos_found"] += count

        # Inspect sample videos
        samples_to_test = video_files[:samples_per_class]
        class_samples_meta = []
        for vid_path in samples_to_test:
            meta = inspect_video_file(vid_path)
            class_samples_meta.append(meta)

        summary["sample_inspections"][class_name] = class_samples_meta

    summary["all_classes_valid"] = (
        len(summary["missing_classes"]) == 0 and summary["total_videos_found"] > 0
    )
    return summary


def save_inspection_summary(summary: Dict[str, Any], output_path: Path | str) -> None:
    """Save the dataset inspection dictionary to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    """CLI entrypoint for dataset inspection."""
    import sys
    from src.utils.config import load_config

    config_path = Path("configs/default_config.yaml")
    config = load_config(config_path)

    raw_dir = config["dataset"]["raw_dir"]
    selected_classes = config["dataset"]["selected_classes"]

    print(f"==================================================")
    print(f"      UCF101 Dataset Inspection (Phase 2)        ")
    print(f"==================================================")
    print(f"Target Directory : {raw_dir}")
    print(f"Selected Classes : {', '.join(selected_classes)}")
    print(f"--------------------------------------------------")

    summary = inspect_dataset(raw_dir, selected_classes, samples_per_class=3)

    if not summary["root_exists"]:
        print(f"[ERROR] Dataset directory not found: {raw_dir}")
        sys.exit(1)

    if summary["missing_classes"]:
        print(f"[WARNING] Missing classes: {summary['missing_classes']}")

    print("\nClass Breakdown:")
    for cls_name, count in summary["class_video_counts"].items():
        print(f"  - {cls_name:<20}: {count:>4} videos")

    print(f"\nTotal Videos Found across 5 classes: {summary['total_videos_found']}")

    print("\nSample Video Readability Check (OpenCV):")
    all_samples_readable = True
    for cls_name, samples in summary["sample_inspections"].items():
        for s in samples:
            status = "OK" if s["valid"] else "FAILED"
            if not s["valid"]:
                all_samples_readable = False
            print(
                f"  [{status}] {s['filename']} -> "
                f"Frames: {s['frame_count']}, FPS: {s['fps']}, Res: {s['width']}x{s['height']}"
            )

    output_file = Path("outputs/dataset_summary.json")
    save_inspection_summary(summary, output_file)
    print(f"\n[SUCCESS] Inspection summary saved to: {output_file.resolve()}")

    if not all_samples_readable:
        print("[ERROR] Some sample videos could not be read!")
        sys.exit(1)


if __name__ == "__main__":
    main()
