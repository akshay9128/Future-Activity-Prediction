"""
Frame Sampler Module.

Loads video files, extracts early observation segments for early action prediction,
uniformly samples a fixed sequence length, resizes, and normalizes frames.
"""

from pathlib import Path
import sys
from typing import Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class VideoLoadingError(Exception):
    """Custom exception raised when a video cannot be opened or decoded."""
    pass


def compute_sample_indices(
    total_frames: int,
    sequence_length: int = 16,
    observation_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute frame indices to sample from the early portion of a video.

    Args:
        total_frames: Total number of frames in the video clip.
        sequence_length: Number of frames to uniformly sample.
        observation_ratio: Ratio of the video from the beginning to observe (0.0 < ratio <= 1.0).
                           e.g., 0.5 means only observe the first 50% of the video.

    Returns:
        np.ndarray: 1D array of integer frame indices.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if not (0.0 < observation_ratio <= 1.0):
        raise ValueError(f"observation_ratio must be in (0.0, 1.0], got {observation_ratio}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    # Maximum frame index within the observed portion
    max_observed_frames = max(1, int(np.floor(total_frames * observation_ratio)))
    
    # Generate uniformly spaced integer indices
    indices = np.linspace(0, max_observed_frames - 1, num=sequence_length, dtype=int)
    return indices


def normalize_frames(frames: np.ndarray, mode: str = "mobilenet_v2") -> np.ndarray:
    """
    Normalize pixel values for neural network consumption.

    Args:
        frames: Array of shape (..., height, width, 3), typically float32 or uint8.
        mode: Normalization strategy:
              - 'mobilenet_v2': maps [0, 255] to [-1.0, 1.0] via (x / 127.5) - 1.0
              - 'zero_one': maps [0, 255] to [0.0, 1.0] via x / 255.0
              - 'none': returns unnormalized float32

    Returns:
        np.ndarray: Normalized array of type np.float32.
    """
    frames_float = frames.astype(np.float32)
    if mode == "mobilenet_v2":
        return (frames_float / 127.5) - 1.0
    elif mode == "zero_one":
        return frames_float / 255.0
    elif mode == "none":
        return frames_float
    else:
        raise ValueError(f"Unknown normalization mode: '{mode}'. Choose 'mobilenet_v2', 'zero_one', or 'none'.")


def load_and_sample_video(
    video_path: Path | str,
    sequence_length: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    observation_ratio: float = 0.5,
    normalization: str = "mobilenet_v2",
) -> np.ndarray:
    """
    Load a video file and sample a normalized sequence of frames from its early portion.

    Args:
        video_path: Path to the video file.
        sequence_length: Desired number of sampled frames (temporal dimension).
        target_size: (height, width) tuple to resize each frame.
        observation_ratio: Fraction of the video from start to observe (e.g. 0.5 for early prediction).
        normalization: Normalization strategy ('mobilenet_v2', 'zero_one', 'none').

    Returns:
        np.ndarray: Array of shape (sequence_length, target_height, target_width, 3) with np.float32.

    Raises:
        VideoLoadingError: If the file does not exist, cannot be opened, or has zero readable frames.
    """
    path = Path(video_path)
    if not path.is_file():
        raise VideoLoadingError(f"Video file does not exist: {path.resolve()}")
    if path.stat().st_size == 0:
        raise VideoLoadingError(f"Video file is empty (0 bytes): {path.resolve()}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoLoadingError(f"OpenCV could not open video: {path.resolve()}")

    target_h, target_w = target_size

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: count frames dynamically if metadata header reports 0
            cap_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                cap_frames += 1
            total_frames = cap_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if total_frames <= 0:
            raise VideoLoadingError(f"Video contains no readable frames: {path.resolve()}")

        # Compute which frame indices to capture
        target_indices = compute_sample_indices(
            total_frames=total_frames,
            sequence_length=sequence_length,
            observation_ratio=observation_ratio,
        )

        # Set of unique indices needed
        needed_indices = set(target_indices)
        max_needed_idx = int(np.max(target_indices))

        # Sequentially decode up to max_needed_idx
        decoded_frames = {}
        current_idx = 0

        while current_idx <= max_needed_idx:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in needed_indices:
                # Convert BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to (target_w, target_h)
                frame_resized = cv2.resize(
                    frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
                decoded_frames[current_idx] = frame_resized
            current_idx += 1

        if len(decoded_frames) == 0:
            raise VideoLoadingError(f"Failed to decode any frames from video: {path.resolve()}")

        # Build output sequence corresponding to target_indices
        available_sorted_indices = sorted(decoded_frames.keys())
        last_available_frame = decoded_frames[available_sorted_indices[-1]]

        sequence = []
        for idx in target_indices:
            if idx in decoded_frames:
                sequence.append(decoded_frames[idx])
            else:
                # Fallback if video ended prematurely before expected index: use closest available frame
                closest_idx = min(available_sorted_indices, key=lambda k: abs(k - idx))
                sequence.append(decoded_frames[closest_idx])

        sequence_arr = np.array(sequence, dtype=np.uint8)

        # Normalize pixel values
        normalized_sequence = normalize_frames(sequence_arr, mode=normalization)
        return normalized_sequence

    finally:
        cap.release()
