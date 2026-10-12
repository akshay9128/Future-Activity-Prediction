"""
Dataset Manifest Module.

Discovers video files across selected action classes, extracts group identifiers,
performs leakage-safe stratified splitting into train, validation, and test sets,
and exports a reproducible dataset manifest CSV.
"""

from pathlib import Path
import re
import sys
from typing import Dict, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config

SUPPORTED_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def extract_group_id(filename: str) -> str:
    """
    Extract the UCF101 recording group ID from the filename.
    e.g. 'v_Basketball_g01_c01.avi' -> 'g01'.
    If the pattern is not matched, returns the filename stem.
    """
    match = re.search(r"_(g\d+)_", filename)
    if match:
        return match.group(1)
    return Path(filename).stem


def build_dataset_manifest(
    dataset_dir: Path | str,
    selected_classes: List[str],
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    videos_per_class_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a dataset manifest with stratified, disjoint train/val/test splits.

    Args:
        dataset_dir: Root directory of UCF101 dataset.
        selected_classes: List of class folder names.
        train_split: Fraction for training (default: 0.70).
        val_split: Fraction for validation (default: 0.15).
        test_split: Fraction for test (default: 0.15).
        seed: Random seed for reproducible splitting.
        videos_per_class_limit: Optional cap on videos per class for rapid prototyping.

    Returns:
        pd.DataFrame: Manifest dataframe with columns:
                      ['video_id', 'video_path', 'class_name', 'class_idx', 'group_id', 'split']

    Raises:
        ValueError: If split fractions do not sum approximately to 1.0 or if classes are missing.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root.resolve()}")

    total_splits = train_split + val_split + test_split
    if not (0.999 <= total_splits <= 1.001):
        raise ValueError(
            f"Split ratios must sum to 1.0, got: {train_split} + {val_split} + {test_split} = {total_splits}"
        )

    # Deterministic mapping: class names sorted alphabetically -> integer index 0..N-1
    sorted_classes = sorted(selected_classes)
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}

    records = []
    for class_name in sorted_classes:
        class_folder = root / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_folder.resolve()}")

        # Discover and sort video files
        video_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        video_files.sort(key=lambda p: p.name)

        if videos_per_class_limit is not None and videos_per_class_limit > 0:
            video_files = video_files[:videos_per_class_limit]

        for vf in video_files:
            records.append({
                "video_id": vf.name,
                "video_path": str(vf.resolve()),
                "class_name": class_name,
                "class_idx": class_to_idx[class_name],
                "group_id": extract_group_id(vf.name),
            })

    if not records:
        raise ValueError(f"No valid video files found in {root} for classes {selected_classes}")

    df = pd.DataFrame(records)

    # Perform Stratified Train / Val / Test Split
    # First split into train vs temp (val + test)
    temp_ratio = val_split + test_split
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["class_idx"],
        random_state=seed,
    )

    # Split temp into val vs test proportionally
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

    # Verify zero data leakage across splits
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

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If required columns are missing.
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


def main() -> None:
    """CLI script to build and save the dataset manifest."""
    config = load_config(Path("configs/default_config.yaml"))

    dataset_cfg = config["dataset"]
    output_cfg = config["paths"]["manifest_csv"]
    seed = config.get("seed", 42)

    print("==================================================")
    print("            Building Dataset Manifest             ")
    print("==================================================")

    print(f"Dataset Root   : {dataset_cfg['raw_dir']}")
    print(f"Selected Classes: {dataset_cfg['selected_classes']}")
    print(f"Splits Target  : Train {dataset_cfg['train_split']*100:.0f}% / Val {dataset_cfg['val_split']*100:.0f}% / Test {dataset_cfg['test_split']*100:.0f}%")
    print(f"Random Seed    : {seed}")
    print("--------------------------------------------------")

    df = build_dataset_manifest(
        dataset_dir=dataset_cfg["raw_dir"],
        selected_classes=dataset_cfg["selected_classes"],
        train_split=dataset_cfg["train_split"],
        val_split=dataset_cfg["val_split"],
        test_split=dataset_cfg["test_split"],
        seed=seed,
        videos_per_class_limit=dataset_cfg.get("videos_per_class_limit"),
    )

    save_manifest(df, output_cfg)
    print(f"[SUCCESS] Manifest saved to: {Path(output_cfg).resolve()}")
    print(f"Total Videos Indexed: {len(df)}")

    # Print summary split distribution
    summary_table = pd.crosstab(df["class_name"], df["split"], margins=True)
    print("\nSplit Distribution by Class:")
    print(summary_table.to_string())


if __name__ == "__main__":
    main()
