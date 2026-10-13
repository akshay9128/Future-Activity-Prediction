"""
Test Suite for Interactive Web Application and Real-Time Detector.

Validates the real-time sliding frame buffer, stream simulation, on-screen HUD
drawing, and Streamlit app helper routines.
"""

from pathlib import Path
import numpy as np
import pytest

from src.app.streamlit_app import get_cached_predictor, get_sample_videos_from_manifest
from src.inference.realtime_detector import RealtimeActivityDetector

SAMPLE_VIDEO_PATH = Path("D:/datasets/UCF-101/Basketball/v_Basketball_g01_c01.avi")


def test_realtime_detector_initialization():
    """Verify initialization of RealtimeActivityDetector with default parameters."""
    detector = RealtimeActivityDetector(
        model_type="cnn_lstm_attention",
        buffer_size=16,
        inference_interval=2,
    )
    assert detector.buffer_size == 16
    assert len(detector.frame_buffer) == 0
    assert len(detector.classes) in (5, 101)
    assert detector.attention_extractor is not None


def test_realtime_detector_sliding_buffer_and_hud():
    """Verify frame ingestion, buffer filling, inference generation, and HUD rendering."""
    detector = RealtimeActivityDetector(
        model_type="cnn_lstm_attention",
        buffer_size=16,
        inference_interval=1,
    )

    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    # Ingest 15 frames: buffer not yet full
    for i in range(15):
        annotated, pred = detector.process_frame(dummy_frame, draw_hud=True)
        assert annotated.shape == (240, 320, 3)
        assert pred is None

    # Ingest 16th frame: buffer reaches 16 frames, trigger inference
    annotated, pred = detector.process_frame(dummy_frame, draw_hud=True)
    assert annotated.shape == (240, 320, 3)
    assert pred is not None
    assert "predicted_class" in pred
    assert "confidence_pct" in pred
    assert pred["predicted_class"] in detector.classes
    assert len(detector.frame_buffer) == 16

    # Ingest 17th frame: FIFO queue discards oldest frame, maintains 16
    annotated, pred2 = detector.process_frame(dummy_frame, draw_hud=True)
    assert len(detector.frame_buffer) == 16
    assert pred2 is not None


def test_realtime_detector_stream_simulation():
    """Verify headless stream processing on a real video file."""
    assert SAMPLE_VIDEO_PATH.exists(), f"Sample video missing: {SAMPLE_VIDEO_PATH}"

    detector = RealtimeActivityDetector(
        model_type="cnn_lstm",
        buffer_size=16,
        inference_interval=2,
    )

    # Run stream headlessly for 20 frames
    detector.run_stream(
        video_source=str(SAMPLE_VIDEO_PATH),
        display=False,
        max_frames=20,
    )

    assert len(detector.frame_buffer) == 16
    assert detector.last_prediction is not None
    assert detector.last_prediction["predicted_class"] in detector.classes


def test_sample_videos_from_manifest_retrieval():
    """Verify get_sample_videos_from_manifest returns verified sample videos."""
    samples = get_sample_videos_from_manifest()
    assert isinstance(samples, dict)
    assert len(samples) in (5, 101), f"Expected 5 or 101 class samples, got {len(samples)}"
    for label, path_str in samples.items():
        assert Path(path_str).exists(), f"Sample video path does not exist: {path_str}"


def test_cached_predictor_creation():
    """Verify get_cached_predictor returns working ActivityPredictor."""
    predictor = get_cached_predictor("cnn_baseline")
    assert predictor.model_type == "cnn_baseline"
    assert len(predictor.classes) in (5, 101)


def test_streamlit_alias_compatibility():
    """Verify both src.app.streamlit and src.app.streamlit_app export the expected functions."""
    import src.app.streamlit as st_module
    import src.app.streamlit_app as st_app_module

    assert hasattr(st_module, "main")
    assert hasattr(st_module, "get_cached_predictor")
    assert hasattr(st_app_module, "main")
    assert hasattr(st_app_module, "get_cached_predictor")
    assert hasattr(st_module, "ensure_web_compatible_video")
    assert hasattr(st_app_module, "ensure_web_compatible_video")


def test_ensure_web_compatible_video_transcoding():
    """Verify ensure_web_compatible_video converts UCF-101 AVI into web-compatible MP4."""
    from src.app.streamlit_app import ensure_web_compatible_video

    assert SAMPLE_VIDEO_PATH.exists()
    web_mp4 = ensure_web_compatible_video(str(SAMPLE_VIDEO_PATH))
    assert Path(web_mp4).exists()
    assert str(web_mp4).endswith(".mp4")
    assert Path(web_mp4).stat().st_size > 0


