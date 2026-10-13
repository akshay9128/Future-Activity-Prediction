"""
Test Suite for Standalone Activity Predictor and Inference Engine.

Validates single-video inference across all three architectures, output schema,
confidence scores, attention keyframe extraction, error handling, and visual dashboard creation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.inference.predict import ActivityPredictor, predict_video_cli

SAMPLE_VIDEO_PATH = Path("D:/datasets/UCF-101/Basketball/v_Basketball_g01_c01.avi")


def test_activity_predictor_initialization():
    """Verify initialization of ActivityPredictor for CNN + LSTM architecture."""
    predictor_default = ActivityPredictor()
    assert predictor_default.model_type == "cnn_lstm"
    assert len(predictor_default.classes) in (5, 101)

    predictor_attn = ActivityPredictor(model_type="cnn_lstm_attention")
    assert predictor_attn.model_type == "cnn_lstm_attention"
    assert len(predictor_attn.classes) in (5, 101)
    assert predictor_attn.attention_extractor is not None

    predictor_lstm = ActivityPredictor(model_type="cnn_lstm")
    assert predictor_lstm.model_type == "cnn_lstm"
    assert predictor_lstm.attention_extractor is None

    predictor_base = ActivityPredictor(model_type="cnn_baseline")
    assert predictor_base.model_type == "cnn_baseline"

    with pytest.raises(ValueError, match="Unsupported model_type"):
        ActivityPredictor(model_type="unsupported_architecture")


def test_missing_checkpoint_raises_file_not_found():
    """Verify that missing model weights raise FileNotFoundError with a clear message."""
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        ActivityPredictor(model_type="cnn_lstm", model_path="outputs/models/nonexistent_weights.keras")


def test_predict_top5_and_latency_schema():
    """Verify top-5 predictions and latency schema on single video using CNN + LSTM."""
    assert SAMPLE_VIDEO_PATH.exists()
    predictor = ActivityPredictor(model_type="cnn_lstm")
    result = predictor.predict_video(SAMPLE_VIDEO_PATH)

    assert "top_5_predictions" in result
    assert len(result["top_5_predictions"]) == min(5, len(predictor.classes))
    for p in result["top_5_predictions"]:
        assert "rank" in p
        assert "class_name" in p
        assert "confidence" in p
        assert "confidence_pct" in p
        assert 0.0 <= p["confidence"] <= 1.0

    assert "inference_time_ms" in result
    assert result["inference_time_ms"] > 0.0
    assert result["dataset_source"] == "held_out_test_set"


def test_predict_single_video_output_schema():
    """Verify that predict_video on real video returns complete prediction dictionary."""
    assert SAMPLE_VIDEO_PATH.exists(), f"Sample video missing: {SAMPLE_VIDEO_PATH}"

    predictor = ActivityPredictor(model_type="cnn_lstm_attention")
    result = predictor.predict_video(SAMPLE_VIDEO_PATH, observation_ratio=0.5)

    assert result["predicted_class"] == "Basketball"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["sequence_length"] == 16
    assert result["observation_ratio"] == 0.5

    # Check probabilities
    class_probs = result["class_probabilities"]
    assert len(class_probs) in (5, 101)
    prob_sum = sum(class_probs.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-2), f"Probabilities do not sum to 1.0: {prob_sum}"

    # Check attention weights
    attn_weights = result["attention_weights"]
    assert attn_weights is not None
    assert len(attn_weights) == 16
    assert np.isclose(sum(attn_weights), 1.0, atol=1e-2)

    top_frames = result["top_attended_frames"]
    assert top_frames is not None
    assert len(top_frames) == 3
    assert all(1 <= f <= 16 for f in top_frames)


def test_predict_all_architectures():
    """Verify prediction capability across CNN Baseline and CNN+LSTM architectures."""
    assert SAMPLE_VIDEO_PATH.exists()

    for m_type in ["cnn_baseline", "cnn_lstm"]:
        predictor = ActivityPredictor(model_type=m_type)
        res = predictor.predict_video(SAMPLE_VIDEO_PATH)

        assert res["predicted_class"] in predictor.classes
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["attention_weights"] is None
        assert res["top_attended_frames"] is None


def test_predict_nonexistent_video_raises_file_not_found():
    """Verify that non-existent video path raises FileNotFoundError."""
    predictor = ActivityPredictor(model_type="cnn_lstm_attention")
    with pytest.raises(FileNotFoundError):
        predictor.predict_video("non_existent_fake_video.avi")


def test_visualize_prediction_dashboard_created(tmp_path: Path):
    """Verify that visualize_prediction generates non-empty image file."""
    assert SAMPLE_VIDEO_PATH.exists()

    predictor = ActivityPredictor(model_type="cnn_lstm_attention")
    result = predictor.predict_video(SAMPLE_VIDEO_PATH, return_frames=True)

    out_img = tmp_path / "test_prediction_dashboard.png"
    predictor.visualize_prediction(
        prediction_result=result,
        display_frames=result["display_frames"],
        output_path=out_img,
    )

    assert out_img.exists()
    assert out_img.stat().st_size > 10_000, "Generated dashboard image is too small"


def test_cli_function_invocation(tmp_path: Path):
    """Verify predict_video_cli runs cleanly with save-viz and output-json."""
    assert SAMPLE_VIDEO_PATH.exists()

    viz_path = tmp_path / "cli_viz.png"
    json_path = tmp_path / "cli_out.json"

    result = predict_video_cli(
        video_path=str(SAMPLE_VIDEO_PATH),
        model_type="cnn_lstm_attention",
        save_viz=str(viz_path),
        output_json=str(json_path),
    )

    assert result["predicted_class"] == "Basketball"
    assert viz_path.exists()
    assert json_path.exists()
