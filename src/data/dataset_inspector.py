"""
Dataset Inspector Module.

Inspects the UCF101 dataset on disk, verifies the presence of target action classes,
counts available videos, checks video readability using OpenCV, detects duplicate
or unreadable video files, and generates a structured summary report.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import cv2

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def discover_classes(dataset_dir: Path | str) -> List[str]:
    """
    Auto-discover all class directories in the UCF101 root folder.

    Args:
        dataset_dir: Root directory containing class folders.

    Returns:
        List[str]: Alphabetically sorted list of class folder names.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        return []

    classes = [
        d.name
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    classes.sort()
    return classes


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
    selected_classes: Optional[Union[List[str], str]] = None,
    samples_per_class: int = 2,
) -> Dict[str, Any]:
    """
    Inspect the UCF101 dataset directory for specified or all action classes.

    Args:
        dataset_dir: Root directory of UCF101 dataset.
        selected_classes: List of class folder names, or 'all'/None to auto-discover all classes.
        samples_per_class: Number of sample videos per class to test for OpenCV readability.

    Returns:
        dict: Inspection summary containing class counts, sample checks, duplicates, and status.
    """
    root = Path(dataset_dir)
    discovered_all = discover_classes(root)

    if selected_classes is None or selected_classes == "all":
        classes_to_check = discovered_all
    elif isinstance(selected_classes, str):
        classes_to_check = [selected_classes]
    else:
        classes_to_check = list(selected_classes)

    summary: Dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "root_exists": root.is_dir(),
        "total_classes_discovered": len(discovered_all),
        "total_classes_requested": len(classes_to_check),
        "available_classes": [],
        "missing_classes": [],
        "empty_classes": [],
        "class_video_counts": {},
        "total_videos_found": 0,
        "duplicate_video_paths": [],
        "sample_inspections": {},
        "unreadable_samples": [],
        "all_classes_valid": False,
    }

    if not root.is_dir():
        summary["error"] = f"Dataset root directory does not exist: {root}"
        return summary

    seen_paths = set()

    for class_name in classes_to_check:
        class_folder = root / class_name
        if not class_folder.is_dir():
            summary["missing_classes"].append(class_name)
            continue

        summary["available_classes"].append(class_name)

        # Collect video files with supported extensions
        video_files = [
            f
            for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        video_files.sort(key=lambda p: p.name)

        count = len(video_files)
        summary["class_video_counts"][class_name] = count
        summary["total_videos_found"] += count

        if count == 0:
            summary["empty_classes"].append(class_name)

        # Check duplicate paths
        for vf in video_files:
            resolved_p = str(vf.resolve())
            if resolved_p in seen_paths:
                summary["duplicate_video_paths"].append(resolved_p)
            seen_paths.add(resolved_p)

        # Inspect sample videos
        samples_to_test = video_files[:samples_per_class] if samples_per_class > 0 else []
        class_samples_meta = []
        for vid_path in samples_to_test:
            meta = inspect_video_file(vid_path)
            class_samples_meta.append(meta)
            if not meta["valid"]:
                summary["unreadable_samples"].append(meta)

        summary["sample_inspections"][class_name] = class_samples_meta

    summary["all_classes_valid"] = (
        len(summary["missing_classes"]) == 0
        and len(summary["empty_classes"]) == 0
        and len(summary["unreadable_samples"]) == 0
        and summary["total_videos_found"] > 0
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
    from src.utils.config import load_config

    config_path = Path("configs/default_config.yaml")
    config = load_config(config_path)

    raw_dir = config["dataset"]["raw_dir"]
    selected_classes = config["dataset"].get("selected_classes", "all")

    print("==================================================")
    print("            UCF101 Dataset Inspection             ")
    print("==================================================")
    print(f"Target Directory : {raw_dir}")
    target_desc = "All 101 Classes" if selected_classes == "all" else f"{len(selected_classes)} Classes"
    print(f"Target Scope     : {target_desc}")
    print("--------------------------------------------------")

    summary = inspect_dataset(raw_dir, selected_classes, samples_per_class=2)

    if not summary["root_exists"]:
        print(f"[ERROR] Dataset directory not found: {raw_dir}")
        sys.exit(1)

    print(f"Total Discovered Classes : {summary['total_classes_discovered']}")
    print(f"Total Videos Indexed     : {summary['total_videos_found']}")
    print(f"Duplicate Video Paths    : {len(summary['duplicate_video_paths'])}")
    print(f"Empty Classes Detected   : {len(summary['empty_classes'])}")
    print(f"Unreadable Sample Videos : {len(summary['unreadable_samples'])}")

    if summary["missing_classes"]:
        print(f"[WARNING] Missing classes ({len(summary['missing_classes'])}): {summary['missing_classes']}")

    if summary["empty_classes"]:
        print(f"[WARNING] Empty classes: {summary['empty_classes']}")

    # Save to both locations
    primary_out = Path("outputs/reports/dataset_inspection_101.json")
    compat_out = Path("outputs/dataset_summary.json")
    save_inspection_summary(summary, primary_out)
    save_inspection_summary(summary, compat_out)
    print(f"\n[SUCCESS] Inspection summary saved to:")
    print(f"  - {primary_out.resolve()}")
    print(f"  - {compat_out.resolve()}")

    if not summary["all_classes_valid"]:
        print("[ERROR] Dataset verification failed! Please resolve missing or corrupted folders.")
        sys.exit(1)
    else:
        print("[SUCCESS] All 101 UCF101 classes verified and validated!")


if __name__ == "__main__":
    main()
