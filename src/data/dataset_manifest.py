"""
Dataset Manifest Module.

Discovers video files across action classes, extracts group identifiers,
performs leakage-safe stratified splitting into train, validation, and test sets,
exports class-to-index mappings, and saves a reproducible dataset manifest CSV.
"""

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_inspector import discover_classes
from src.utils.config import load_config

SUPPORTED_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def extract_group_id(filename: str) -> str:
    """
    Extract the UCF101 recording group ID from the filename.
    e.g. 'v_Basketball_g01_c01.avi' -> 'g01'.
    If pattern is not matched, returns the filename stem.
    """
    match = re.search(r"_(g\d+)_", filename)
    if match:
        return match.group(1)
    return Path(filename).stem


def build_dataset_manifest(
    dataset_dir: Path | str,
    selected_classes: Optional[Union[List[str], str]] = "all",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    videos_per_class_limit: Optional[int] = None,
    class_to_idx_output: Optional[Path | str] = "outputs/class_to_idx.json",
    idx_to_class_output: Optional[Path | str] = "outputs/idx_to_class.json",
) -> pd.DataFrame:
    """
    Create a dataset manifest with stratified, disjoint train/val/test splits
    and save stable class-to-index mappings.

    Args:
        dataset_dir: Root directory of UCF101 dataset.
        selected_classes: List of class folder names, or 'all'/None to use all discovered classes.
        train_split: Fraction for training (default: 0.70).
        val_split: Fraction for validation (default: 0.15).
        test_split: Fraction for test (default: 0.15).
        seed: Random seed for reproducible splitting.
        videos_per_class_limit: Optional cap on videos per class for rapid prototyping.
        class_to_idx_output: Path to save class_to_idx JSON mapping.
        idx_to_class_output: Path to save idx_to_class JSON mapping.

    Returns:
        pd.DataFrame: Manifest dataframe with columns:
                      ['video_id', 'video_path', 'class_name', 'class_idx', 'group_id', 'split']
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root.resolve()}")

    total_splits = train_split + val_split + test_split
    if not (0.999 <= total_splits <= 1.001):
        raise ValueError(
            f"Split ratios must sum to 1.0, got: {train_split} + {val_split} + {test_split} = {total_splits}"
        )

    # 1. Class Discovery
    if selected_classes is None or selected_classes == "all":
        classes_to_use = discover_classes(root)
    elif isinstance(selected_classes, str):
        classes_to_use = [selected_classes]
    else:
        classes_to_use = sorted(list(selected_classes))

    if not classes_to_use:
        raise ValueError(f"No class directories found in {root.resolve()}")

    # 2. Stable Alphabetical Integer Mapping: 0 to N-1
    sorted_classes = sorted(classes_to_use)
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(sorted_classes)}
    idx_to_class: Dict[str, str] = {str(idx): name for idx, name in enumerate(sorted_classes)}

    # Save mapping files
    if class_to_idx_output:
        p_c2i = Path(class_to_idx_output)
        p_c2i.parent.mkdir(parents=True, exist_ok=True)
        with open(p_c2i, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

    if idx_to_class_output:
        p_i2c = Path(idx_to_class_output)
        p_i2c.parent.mkdir(parents=True, exist_ok=True)
        with open(p_i2c, "w", encoding="utf-8") as f:
            json.dump(idx_to_class, f, indent=2)

    # 3. Collect video records
    records = []
    seen_video_paths = set()
    duplicate_paths = []

    for class_name in sorted_classes:
        class_folder = root / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_folder.resolve()}")

        video_files = [
            f
            for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        video_files.sort(key=lambda p: p.name)

        if videos_per_class_limit is not None and videos_per_class_limit > 0:
            video_files = video_files[:videos_per_class_limit]

        for vf in video_files:
            resolved_p = str(vf.resolve())
            if resolved_p in seen_video_paths:
                duplicate_paths.append(resolved_p)
            seen_video_paths.add(resolved_p)

            records.append({
                "video_id": vf.name,
                "video_path": resolved_p,
                "class_name": class_name,
                "class_idx": class_to_idx[class_name],
                "group_id": extract_group_id(vf.name),
            })

    if duplicate_paths:
        raise ValueError(f"Found {len(duplicate_paths)} duplicate video paths across dataset!")

    if not records:
        raise ValueError(f"No valid video files found in {root} for classes {classes_to_use}")

    df = pd.DataFrame(records)

    # 4. Perform Stratified Train / Val / Test Split
    temp_ratio = val_split + test_split
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["class_idx"],
        random_state=seed,
    )

    val_rel_ratio = val_split / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df["class_idx"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values(by=["class_idx", "video_id"]).reset_index(drop=True)

    # 5. Verify Zero Data Leakage Across Splits
    train_paths = set(train_df["video_path"])
    val_paths = set(val_df["video_path"])
    test_paths = set(test_df["video_path"])

    assert len(train_paths & val_paths) == 0, "Data leakage detected: train and val share videos!"
    assert len(train_paths & test_paths) == 0, "Data leakage detected: train and test share videos!"
    assert len(val_paths & test_paths) == 0, "Data leakage detected: val and test share videos!"

    return combined


def save_manifest(df: pd.DataFrame, output_path: Path | str) -> None:
    """Save the manifest DataFrame to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """
    Load a dataset manifest CSV and validate required columns.

    Args:
        manifest_path: Path to the CSV manifest file.

    Returns:
        pd.DataFrame: Loaded manifest.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path.resolve()}")

    df = pd.read_csv(path)
    required_cols = {"video_id", "video_path", "class_name", "class_idx", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    return df


def save_split_summary_report(df: pd.DataFrame, output_path: Path | str) -> Dict[str, Any]:
    """Compile and save split summary statistics report to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    split_counts = df["split"].value_counts().to_dict()
    num_classes = df["class_idx"].nunique()

    per_class_splits = {}
    for c_name, c_group in df.groupby("class_name"):
        per_class_splits[c_name] = c_group["split"].value_counts().to_dict()

    summary = {
        "total_videos": int(len(df)),
        "total_classes": int(num_classes),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "split_percentages": {
            k: round((v / len(df)) * 100, 2) for k, v in split_counts.items()
        },
        "per_class_splits": per_class_splits,
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """CLI script to build and save the dataset manifest."""
    config = load_config(Path("configs/default_config.yaml"))

    dataset_cfg = config["dataset"]
    output_cfg = config["paths"]["manifest_csv"]
    c2i_out = config["paths"].get("class_to_idx_json", "outputs/class_to_idx.json")
    i2c_out = config["paths"].get("idx_to_class_json", "outputs/idx_to_class.json")
    seed = config.get("seed", 42)

    print("==================================================")
    print("            Building Dataset Manifest             ")
    print("==================================================")
    print(f"Dataset Root    : {dataset_cfg['raw_dir']}")
    scope = "All 101 Classes" if dataset_cfg.get("selected_classes") == "all" else f"{len(dataset_cfg['selected_classes'])} Classes"
    print(f"Target Scope    : {scope}")
    print(f"Splits Target   : Train {dataset_cfg['train_split']*100:.0f}% / Val {dataset_cfg['val_split']*100:.0f}% / Test {dataset_cfg['test_split']*100:.0f}%")
    print(f"Random Seed     : {seed}")
    print("--------------------------------------------------")

    df = build_dataset_manifest(
        dataset_dir=dataset_cfg["raw_dir"],
        selected_classes=dataset_cfg.get("selected_classes", "all"),
        train_split=dataset_cfg["train_split"],
        val_split=dataset_cfg["val_split"],
        test_split=dataset_cfg["test_split"],
        seed=seed,
        videos_per_class_limit=dataset_cfg.get("videos_per_class_limit"),
        class_to_idx_output=c2i_out,
        idx_to_class_output=i2c_out,
    )

    save_manifest(df, output_cfg)
    report_path = Path("outputs/reports/dataset_split_summary_101.json")
    summary = save_split_summary_report(df, report_path)

    print(f"[SUCCESS] Manifest saved to: {Path(output_cfg).resolve()}")
    print(f"[SUCCESS] Class Mapping saved to: {Path(c2i_out).resolve()}")
    print(f"[SUCCESS] Split Report saved to: {report_path.resolve()}")
    print(f"\nTotal Videos Indexed : {len(df)}")
    print(f"Total Unique Classes : {df['class_idx'].nunique()}")
    print(f"Train Split          : {summary['split_counts'].get('train', 0)} ({summary['split_percentages'].get('train', 0):.1f}%)")
    print(f"Val Split            : {summary['split_counts'].get('val', 0)} ({summary['split_percentages'].get('val', 0):.1f}%)")
    print(f"Test Split           : {summary['split_counts'].get('test', 0)} ({summary['split_percentages'].get('test', 0):.1f}%)")


if __name__ == "__main__":
    main()
