"""
Feature Caching Module.

Extracts CNN spatial embeddings for video clips across dataset splits and caches them
to compressed .npz archives, drastically accelerating downstream model training
(CNN baseline, LSTM, Temporal Attention) without redundant CNN passes.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manifest import load_manifest
from src.data.frame_sampler import load_and_sample_video, VideoLoadingError
from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.utils.config import load_config


def validate_cache_archive(
    cache_path: Path | str,
    expected_samples: Optional[int] = None,
    expected_seq_len: int = 16,
    expected_feat_dim: int = 1280,
) -> Dict[str, Any]:
    """
    Validate the structural and numeric integrity of a cached .npz feature file.

    Args:
        cache_path: Path to .npz file.
        expected_samples: Expected sample count, or None to accept any positive count.
        expected_seq_len: Expected temporal sequence length (default 16).
        expected_feat_dim: Expected feature embedding dimension (default 1280).

    Returns:
        Dict[str, Any]: Status summary containing 'valid', 'samples', 'seq_len', etc.
    """
    path = Path(cache_path)
    result: Dict[str, Any] = {
        "valid": False,
        "path": str(path),
        "samples": 0,
        "sequence_length": 0,
        "feature_dim": 0,
        "num_classes": 0,
        "error": None,
    }

    if not path.is_file():
        result["error"] = "File does not exist"
        return result

    try:
        data = np.load(path)
        if "features" not in data or "labels" not in data:
            result["error"] = "Missing required keys 'features' or 'labels'"
            return result

        features = data["features"]
        labels = data["labels"]

        if len(features.shape) != 3:
            result["error"] = f"Expected 3D features (N, T, D), got shape {features.shape}"
            return result

        n_samples, seq_len, feat_dim = features.shape
        result["samples"] = int(n_samples)
        result["sequence_length"] = int(seq_len)
        result["feature_dim"] = int(feat_dim)
        result["num_classes"] = int(len(np.unique(labels)))

        if expected_samples is not None and n_samples != expected_samples:
            result["error"] = f"Sample count mismatch: expected {expected_samples}, found {n_samples}"
            return result

        if seq_len != expected_seq_len:
            result["error"] = f"Sequence length mismatch: expected {expected_seq_len}, found {seq_len}"
            return result

        if feat_dim != expected_feat_dim:
            result["error"] = f"Feature dim mismatch: expected {expected_feat_dim}, found {feat_dim}"
            return result

        if len(labels) != n_samples:
            result["error"] = f"Label length ({len(labels)}) does not match features ({n_samples})"
            return result

        if not np.all(np.isfinite(features[:min(10, n_samples)])):
            result["error"] = "Features contain non-finite values (NaN or Inf)"
            return result

        result["valid"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def get_or_extract_split_features(
    manifest_df: pd.DataFrame,
    split_name: str,
    feature_extractor: CNNFeatureExtractor,
    cache_dir: Path | str = "outputs/cache",
    sequence_length: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    observation_ratio: float = 0.5,
    batch_size: int = 16,
    force_recompute: bool = False,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached CNN features for a split, or extract and save them if not cached.

    Args:
        manifest_df: Dataset manifest DataFrame.
        split_name: 'train', 'val', or 'test'.
        feature_extractor: Initialized CNNFeatureExtractor instance.
        cache_dir: Directory where .npz archives are saved.
        sequence_length: Number of frames per video.
        target_size: (H, W) frame resolution.
        observation_ratio: Fraction of video observed from the beginning.
        batch_size: Batch size for CNN forward passes.
        force_recompute: Whether to ignore existing cache.
        max_samples_per_class: Optional max videos per class for balanced subset caching.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: (N, sequence_length, feature_dim) float32 array
            - labels: (N,) int64 array
    """
    cache_path = Path(cache_dir) / f"{split_name}_features.npz"
    metadata_path = Path(cache_dir) / f"{split_name}_metadata.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    split_df = manifest_df[manifest_df["split"] == split_name].reset_index(drop=True)
    if max_samples_per_class is not None and max_samples_per_class > 0:
        split_df = (
            split_df.groupby("class_idx", group_keys=False)
            .head(max_samples_per_class)
            .reset_index(drop=True)
        )
    expected_count = len(split_df)

    # 1. Check existing cache
    if cache_path.is_file() and not force_recompute:
        check = validate_cache_archive(
            cache_path=cache_path,
            expected_samples=expected_count,
            expected_seq_len=sequence_length,
            expected_feat_dim=feature_extractor.feature_dim,
        )
        if check["valid"]:
            print(
                f"[CACHE] Loading '{split_name}' features from {cache_path} "
                f"({check['samples']} samples, {check['num_classes']} classes)...",
                flush=True,
            )
            data = np.load(cache_path)
            return data["features"], data["labels"]
        else:
            print(
                f"[CACHE] Existing cache for '{split_name}' invalid ({check['error']}). Recomputing...",
                flush=True,
            )

    # 2. Extract features
    print(
        f"[EXTRACT] Extracting CNN features for '{split_name}' split ({expected_count} videos, batch_size={batch_size})...",
        flush=True,
    )

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []

    batch_sequences: List[np.ndarray] = []
    batch_labels: List[int] = []

    start_time = time.time()
    for i, row in split_df.iterrows():
        vid_path = row["video_path"]
        class_idx = int(row["class_idx"])

        try:
            raw_seq = load_and_sample_video(
                video_path=vid_path,
                sequence_length=sequence_length,
                target_size=target_size,
                observation_ratio=observation_ratio,
                normalization="mobilenet_v2",
            )
            batch_sequences.append(raw_seq)
            batch_labels.append(class_idx)
        except VideoLoadingError as e:
            print(f"[WARN] Skipping corrupted video {vid_path}: {e}", flush=True)

        # Process batch when full or at end of dataset
        if len(batch_sequences) == batch_size or (i + 1 == expected_count and len(batch_sequences) > 0):
            seq_array = np.array(batch_sequences, dtype=np.float32)
            feat_array = feature_extractor.extract_sequence_features(seq_array)
            for feat, lbl in zip(feat_array, batch_labels):
                all_features.append(feat)
                all_labels.append(lbl)
            batch_sequences.clear()
            batch_labels.clear()

        if (i + 1) % 100 == 0 or (i + 1) == expected_count:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 0.001)
            remaining = (expected_count - (i + 1)) / max(rate, 0.001)
            print(
                f"  [{split_name}] Processed {i + 1}/{expected_count} videos "
                f"({rate:.1f} vids/s, ETA: {remaining:.0f}s)...",
                flush=True,
            )

    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # 3. Save compressed .npz archive
    np.savez_compressed(cache_path, features=features_arr, labels=labels_arr)
    print(f"[SAVED] Cached {len(features_arr)} {split_name} samples to {cache_path}", flush=True)

    # 4. Save metadata JSON
    meta_info = {
        "split": split_name,
        "num_samples": int(len(features_arr)),
        "sequence_length": int(sequence_length),
        "feature_dim": int(feature_extractor.feature_dim),
        "target_size": list(target_size),
        "observation_ratio": float(observation_ratio),
        "num_classes": int(len(np.unique(labels_arr))),
        "cache_file": str(cache_path.name),
        "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"[SAVED] Cached metadata saved to {metadata_path}", flush=True)

    return features_arr, labels_arr


def main() -> None:
    """CLI utility for precomputing and verifying feature caches."""
    parser = argparse.ArgumentParser(description="Extract and cache CNN features for UCF-101 splits.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Split to extract.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for CNN feature extraction.")
    parser.add_argument("--force", action="store_true", help="Force recomputing existing caches.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max videos per class for balanced subset.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(config["paths"]["manifest_csv"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run dataset_manifest.py first.")

    manifest_df = load_manifest(manifest_path)
    seq_len = config["preprocessing"]["sequence_length"]
    target_size = tuple(config["preprocessing"]["target_size"])
    obs_ratio = config["preprocessing"]["observation_ratio"]
    cache_dir = Path("outputs/cache")

    extractor = CNNFeatureExtractor(
        backbone_name=config["model"]["backbone"],
        input_shape=(*target_size, 3),
        trainable=False,
        sequence_length=seq_len,
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        print(f"\nProcessing split: {sp}")
        max_samples = args.max_per_class
        if max_samples is None:
            if sp == "train":
                max_samples = config["dataset"].get("max_train_per_class", None)
            elif sp == "val":
                max_samples = config["dataset"].get("max_val_per_class", None)
            elif sp == "test":
                max_samples = config["dataset"].get("max_test_per_class", None)

        get_or_extract_split_features(
            manifest_df=manifest_df,
            split_name=sp,
            feature_extractor=extractor,
            cache_dir=cache_dir,
            sequence_length=seq_len,
            target_size=target_size,
            observation_ratio=obs_ratio,
            batch_size=args.batch_size,
            force_recompute=args.force,
            max_samples_per_class=max_samples,
        )


if __name__ == "__main__":
    main()
