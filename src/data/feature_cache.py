"""
Feature Caching Module.

Extracts CNN spatial embeddings for video clips across dataset splits and caches them
to compressed .npz archives, drastically accelerating downstream model training
(CNN baseline, LSTM, Temporal Attention) without redundant CNN passes.
"""

from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frame_sampler import load_and_sample_video, VideoLoadingError
from src.models.cnn_feature_extractor import CNNFeatureExtractor


def get_or_extract_split_features(
    manifest_df: pd.DataFrame,
    split_name: str,
    feature_extractor: CNNFeatureExtractor,
    cache_dir: Path | str = "outputs/cache",
    sequence_length: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    observation_ratio: float = 0.5,
    force_recompute: bool = False,
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
        force_recompute: Whether to ignore existing cache.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: (N, sequence_length, feature_dim) float32 array
            - labels: (N,) int64 array
    """
    cache_path = Path(cache_dir) / f"{split_name}_features.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file() and not force_recompute:
        print(f"[CACHE] Loading {split_name} features from {cache_path}...", flush=True)
        data = np.load(cache_path)
        return data["features"], data["labels"]

    print(f"[EXTRACT] Extracting CNN features for '{split_name}' split...", flush=True)
    split_df = manifest_df[manifest_df["split"] == split_name].reset_index(drop=True)

    all_features = []
    all_labels = []

    total_videos = len(split_df)
    for i, row in split_df.iterrows():
        vid_path = row["video_path"]
        class_idx = int(row["class_idx"])

        try:
            # 1. Sample frames (16, 224, 224, 3) in [-1, 1]
            raw_seq = load_and_sample_video(
                video_path=vid_path,
                sequence_length=sequence_length,
                target_size=target_size,
                observation_ratio=observation_ratio,
                normalization="mobilenet_v2",
            )
            # 2. Extract CNN features (16, 1280)
            feat_seq = feature_extractor.extract_sequence_features(raw_seq)
            all_features.append(feat_seq)
            all_labels.append(class_idx)
        except VideoLoadingError as e:
            print(f"[WARN] Skipping corrupted video {vid_path}: {e}", flush=True)

        if (i + 1) % 50 == 0 or (i + 1) == total_videos:
            print(f"  Processed {i + 1}/{total_videos} videos...", flush=True)

    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Save to disk
    np.savez_compressed(cache_path, features=features_arr, labels=labels_arr)
    print(f"[SAVED] Cached {len(features_arr)} {split_name} samples to {cache_path}", flush=True)


    return features_arr, labels_arr
