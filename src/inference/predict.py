"""
Activity Prediction Inference Module.

Provides standalone single-video inference for early human activity prediction:
- Supports all three trained architectures (CNN Baseline, CNN + LSTM, CNN + LSTM + Attention).
- Samples uniform frames from the early observation portion (default first 50%).
- Extracts class probabilities, confidence scores, and ranking.
- Extracts temporal attention weights to identify the most salient keyframes.
- Generates visual prediction dashboards and summary reports.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frame_sampler import load_and_sample_video, VideoLoadingError
from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm_attention import get_attention_extraction_model
from src.models.temporal_attention import TemporalAttention
from src.utils.config import load_config

DEFAULT_MODEL_PATHS = {
    "cnn_baseline": Path("outputs/models/cnn_baseline_head.keras"),
    "cnn_lstm": Path("outputs/models/cnn_lstm_head.keras"),
    "cnn_lstm_attention": Path("outputs/models/cnn_lstm_attention_head.keras"),
}


class ActivityPredictor:
    """
    Inference engine for predicting human activities from raw video inputs.
    """

    def __init__(
        self,
        model_type: str = "cnn_lstm_attention",
        model_path: Optional[str | Path] = None,
        config_path: str | Path = "configs/default_config.yaml",
    ) -> None:
        """
        Initialize the predictor with feature extractor and classification head.

        Args:
            model_type: 'cnn_lstm_attention', 'cnn_lstm', or 'cnn_baseline'.
            model_path: Path to .keras model weights. Defaults to outputs/models/.
            config_path: Path to default_config.yaml.
        """
        self.config = load_config(config_path)
        self.classes = sorted(self.config["dataset"]["selected_classes"])
        self.model_type = model_type.lower()

        if self.model_type not in DEFAULT_MODEL_PATHS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose from {list(DEFAULT_MODEL_PATHS.keys())}"
            )

        # Resolve model path
        if model_path is None:
            self.model_path = DEFAULT_MODEL_PATHS[self.model_type]
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path.resolve()}")

        # 1. Initialize CNN Feature Extractor
        seq_len = self.config["preprocessing"]["sequence_length"]
        target_size = tuple(self.config["preprocessing"]["target_size"])
        self.feature_extractor = CNNFeatureExtractor(
            backbone_name=self.config["model"]["backbone"],
            input_shape=(*target_size, 3),
            trainable=False,
            sequence_length=seq_len,
        )

        # 2. Load Head Model
        self.head_model = tf.keras.models.load_model(self.model_path, compile=False)

        # 3. If attention model, prepare attention extraction graph
        if self.model_type == "cnn_lstm_attention":
            self.attention_extractor = get_attention_extraction_model(self.head_model)
        else:
            self.attention_extractor = None

    def predict_video(
        self,
        video_path: str | Path,
        observation_ratio: Optional[float] = None,
        return_frames: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute activity prediction on a single video file.

        Args:
            video_path: Path to video file (.avi, .mp4, etc.).
            observation_ratio: Fraction of video duration observed (e.g. 0.5).
            return_frames: If True, includes sampled frames (uint8 RGB) in return dict.

        Returns:
            Dict[str, Any]: Prediction results containing probabilities and attention weights.
        """
        vid_p = Path(video_path)
        if not vid_p.is_file():
            raise FileNotFoundError(f"Video file does not exist: {vid_p.resolve()}")

        seq_len = self.config["preprocessing"]["sequence_length"]
        target_size = tuple(self.config["preprocessing"]["target_size"])
        if observation_ratio is None:
            observation_ratio = float(self.config["preprocessing"]["observation_ratio"])

        # 1. Sample video frames (normalized to [-1, 1] for MobileNetV2)
        normalized_frames = load_and_sample_video(
            video_path=vid_p,
            sequence_length=seq_len,
            target_size=target_size,
            observation_ratio=observation_ratio,
            normalization="mobilenet_v2",
        )

        # Also sample in [0, 1] for visualization if requested
        display_frames = None
        if return_frames:
            display_frames = (normalized_frames + 1.0) / 2.0  # map [-1, 1] back to [0, 1]
            display_frames = np.clip(display_frames, 0.0, 1.0)

        # 2. Extract CNN feature representation (1, 16, 1280)
        feature_sequence = self.feature_extractor.extract_sequence_features(normalized_frames)
        feature_batch = np.expand_dims(feature_sequence, axis=0)

        # 3. Model Inference
        attention_weights: Optional[List[float]] = None
        top_frames: Optional[List[int]] = None

        if self.model_type == "cnn_lstm_attention" and self.attention_extractor is not None:
            probs_arr, weights_arr = self.attention_extractor.predict(feature_batch, verbose=0)
            probs = probs_arr[0]
            attn_np = np.squeeze(weights_arr[0])  # (16,)
            attention_weights = [round(float(w), 4) for w in attn_np]
            # Rank frames by highest attention
            top_indices = np.argsort(attn_np)[::-1][:3]
            top_frames = [int(i + 1) for i in top_indices]  # 1-indexed for display
        else:
            probs = self.head_model.predict(feature_batch, verbose=0)[0]

        # 4. Process Class Probabilities
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {
            cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(self.classes)
        }

        result: Dict[str, Any] = {
            "video_path": str(vid_p.resolve()),
            "video_name": vid_p.name,
            "model_type": self.model_type,
            "observation_ratio": observation_ratio,
            "sequence_length": seq_len,
            "predicted_class": pred_class,
            "predicted_class_index": pred_idx,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 2),
            "class_probabilities": class_probs,
            "attention_weights": attention_weights,
            "top_attended_frames": top_frames,
        }

        if return_frames:
            result["display_frames"] = display_frames

        return result

    def visualize_prediction(
        self,
        prediction_result: Dict[str, Any],
        display_frames: np.ndarray,
        output_path: Path | str,
    ) -> None:
        """
        Generate and save a visual summary dashboard with sampled frames,
        attention weights bar, and class prediction probability breakdown.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        seq_len = prediction_result["sequence_length"]
        attn_weights = prediction_result.get("attention_weights")
        class_probs = prediction_result["class_probabilities"]
        pred_class = prediction_result["predicted_class"]
        confidence = prediction_result["confidence_pct"]

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.2, 1.4], hspace=0.45)

        # 1. Sampled Frame Strip (16 frames)
        frame_ax = fig.add_subplot(gs[0])
        frame_ax.axis("off")
        frame_ax.set_title(
            f"Observed Video Frames (1 to {seq_len}, First {int(prediction_result['observation_ratio']*100)}% of Video)",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        # Create subplots inside the top row for individual frame thumbnails
        sub_gs = gs[0].subgridspec(1, seq_len, wspace=0.08)
        for i in range(seq_len):
            ax = fig.add_subplot(sub_gs[0, i])
            ax.imshow(display_frames[i])
            ax.set_xticks([])
            ax.set_yticks([])
            label = f"T={i+1}"
            if attn_weights is not None:
                is_top = (i + 1) in (prediction_result.get("top_attended_frames") or [])
                color = "red" if is_top else "black"
                ax.set_xlabel(label, fontsize=8, color=color, fontweight="bold" if is_top else "normal")
            else:
                ax.set_xlabel(label, fontsize=8)

        # 2. Attention Weight Distribution (if available)
        if attn_weights is not None:
            attn_ax = fig.add_subplot(gs[1])
            x_steps = np.arange(1, seq_len + 1)
            bars = attn_ax.bar(
                x_steps,
                attn_weights,
                color="#7570b3",
                edgecolor="black",
                width=0.6,
                label="Temporal Attention Weight",
            )
            # Highlight top attended frames
            top_frames = prediction_result.get("top_attended_frames") or []
            for bar, step in zip(bars, x_steps):
                if step in top_frames:
                    bar.set_color("#d95f02")
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.2)

            attn_ax.axhline(1.0 / seq_len, color="gray", linestyle="--", alpha=0.7, label=f"Uniform Baseline ({1.0/seq_len:.4f})")
            attn_ax.set_xticks(x_steps)
            attn_ax.set_xlabel("Sampled Frame Time Step (T)", fontsize=10)
            attn_ax.set_ylabel("Attention Score", fontsize=10)
            attn_ax.set_title("Learned Frame Importance Weights (Attention)", fontsize=11, fontweight="bold", loc="left")
            attn_ax.legend(loc="upper right", fontsize=8)
            attn_ax.grid(axis="y", linestyle=":", alpha=0.5)

        # 3. Class Probability Bar Chart
        prob_ax = fig.add_subplot(gs[2])
        classes = list(class_probs.keys())
        probs = [class_probs[c] * 100 for c in classes]
        colors = ["#2ca02c" if c == pred_class else "#1f77b4" for c in classes]

        bars_p = prob_ax.barh(classes, probs, color=colors, edgecolor="black", height=0.55)
        prob_ax.set_xlabel("Predicted Probability (%)", fontsize=10)
        prob_ax.set_xlim([0, 105])
        prob_ax.set_title(
            f"Prediction: {pred_class} (Confidence: {confidence:.2f}%) | Model: {prediction_result['model_type']}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        prob_ax.grid(axis="x", linestyle=":", alpha=0.6)

        for bar in bars_p:
            w = bar.get_width()
            prob_ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

        plt.suptitle(
            f"Early Activity Prediction: {prediction_result['video_name']}",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.45)
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Saved prediction dashboard to: {out_p.resolve()}", flush=True)

    def print_prediction_summary(self, result: Dict[str, Any]) -> None:
        """Print clean ASCII summary report of the prediction to the console."""
        print("\n" + "=" * 60)
        print("          ACTIVITY PREDICTION INFERENCE REPORT          ")
        print("=" * 60)
        print(f"Video File       : {result['video_name']}")
        print(f"Model Engine     : {result['model_type']}")
        print(f"Observation Ratio: {int(result['observation_ratio'] * 100)}% of video duration")
        print(f"Sampled Frames   : {result['sequence_length']} frames")
        print("-" * 60)
        print(f"PREDICTED ACTION : {result['predicted_class'].upper()}")
        print(f"CONFIDENCE       : {result['confidence_pct']:.2f}%")
        print("-" * 60)
        print("Class Probabilities:")
        for cls_name, prob in result["class_probabilities"].items():
            bar_len = int(prob * 30)
            bar_str = "=" * bar_len + "-" * (30 - bar_len)
            marker = "<-- [PREDICTED]" if cls_name == result["predicted_class"] else ""
            print(f"  {cls_name:<16} : [{bar_str}] {prob * 100:>5.1f}% {marker}")

        if result.get("attention_weights") is not None:
            print("-" * 60)
            top_frames = result.get("top_attended_frames") or []
            print(f"Top Attended Keyframes: {', '.join(f'T={f}' for f in top_frames)}")
            attn_str = " ".join(f"{w:.2f}" for w in result["attention_weights"])
            print(f"Attention Weights     : [{attn_str}]")
        print("=" * 60 + "\n")


def predict_video_cli(
    video_path: str,
    model_type: str = "cnn_lstm_attention",
    model_path: Optional[str] = None,
    observation_ratio: Optional[float] = None,
    save_viz: Optional[str] = None,
    output_json: Optional[str] = None,
    config_path: str = "configs/default_config.yaml",
) -> Dict[str, Any]:
    """
    Convenience function for CLI and programmatic execution.
    """
    predictor = ActivityPredictor(
        model_type=model_type,
        model_path=model_path,
        config_path=config_path,
    )

    return_frames = save_viz is not None
    result = predictor.predict_video(
        video_path=video_path,
        observation_ratio=observation_ratio,
        return_frames=return_frames,
    )

    predictor.print_prediction_summary(result)

    if save_viz is not None and "display_frames" in result:
        predictor.visualize_prediction(
            prediction_result=result,
            display_frames=result["display_frames"],
            output_path=save_viz,
        )

    if output_json is not None:
        out_j = Path(output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        # Exclude raw image arrays from JSON output
        clean_res = {k: v for k, v in result.items() if k != "display_frames"}
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(clean_res, f, indent=2)
        print(f"[SAVED] Saved inference JSON to: {out_j.resolve()}", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict human activity from an early video segment."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (.avi, .mp4, etc.).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lstm_attention",
        choices=["cnn_lstm_attention", "cnn_lstm", "cnn_baseline"],
        help="Architecture to use for inference (default: cnn_lstm_attention).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional custom path to .keras model weights.",
    )
    parser.add_argument(
        "--observation-ratio",
        type=float,
        default=None,
        help="Fraction of video observed from the beginning (default: 0.5 from config).",
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Path to save visual dashboard plot (.png).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save prediction result JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file.",
    )

    args = parser.parse_args()
    predict_video_cli(
        video_path=args.video,
        model_type=args.model,
        model_path=args.weights,
        observation_ratio=args.observation_ratio,
        save_viz=args.save_viz,
        output_json=args.output_json,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
