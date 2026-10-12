"""
Video Dataset and Batch Generator Module.

Provides memory-efficient, on-the-fly batch generation for video frame sequences,
preventing RAM exhaustion while streaming samples to deep learning models.
"""

from pathlib import Path
import sys
from typing import Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frame_sampler import load_and_sample_video, VideoLoadingError


class VideoSequenceDataset:
    """
    On-demand dataset loader that samples video frames without storing all videos in RAM.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: Optional[str] = None,
        sequence_length: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        observation_ratio: float = 0.5,
        normalization: str = "mobilenet_v2",
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            manifest_df: DataFrame loaded from dataset_manifest.csv.
            split: Filter to 'train', 'val', or 'test'. If None, uses all rows.
            sequence_length: Number of frames per sequence.
            target_size: (height, width) frame resolution.
            observation_ratio: Fraction of video observed (e.g. 0.5).
            normalization: Frame normalization strategy ('mobilenet_v2', 'zero_one').
            shuffle: Whether to shuffle samples before iteration.
            seed: Random seed for shuffling.
        """
        if split is not None:
            self.df = manifest_df[manifest_df["split"] == split].copy().reset_index(drop=True)
        else:
            self.df = manifest_df.copy().reset_index(drop=True)

        self.sequence_length = sequence_length
        self.target_size = target_size
        self.observation_ratio = observation_ratio
        self.normalization = normalization
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def get_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Load and sample a single video sequence and its integer class label."""
        row = self.df.iloc[index]
        video_path = row["video_path"]
        class_idx = int(row["class_idx"])

        sequence = load_and_sample_video(
            video_path=video_path,
            sequence_length=self.sequence_length,
            target_size=self.target_size,
            observation_ratio=self.observation_ratio,
            normalization=self.normalization,
        )
        return sequence, class_idx

    def get_batch(self, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch a single batch of sequences and labels (useful for testing and inspection).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sequences: (B, T, H, W, C) float32 array
                - labels: (B,) int64 array
        """
        batch_size = min(batch_size, len(self))
        batch_indices = self.rng.choice(len(self), size=batch_size, replace=False)

        sequences = []
        labels = []
        for idx in batch_indices:
            seq, label = self.get_sample(int(idx))
            sequences.append(seq)
            labels.append(label)

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def iter_batches(self, batch_size: int = 8) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield batches across the dataset for one complete epoch.
        """
        indices = np.copy(self.indices)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            current_batch_indices = indices[start_idx:end_idx]

            batch_sequences = []
            batch_labels = []

            for idx in current_batch_indices:
                try:
                    seq, label = self.get_sample(int(idx))
                    batch_sequences.append(seq)
                    batch_labels.append(label)
                except VideoLoadingError as e:
                    # In case of rare corrupted video, skip and log rather than crashing epoch
                    continue

            if batch_sequences:
                yield (
                    np.array(batch_sequences, dtype=np.float32),
                    np.array(batch_labels, dtype=np.int64),
                )
