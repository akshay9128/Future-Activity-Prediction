"""
Real-Time Streaming Activity Detection Engine.

Provides live webcam and streaming video detection using a rolling FIFO buffer of 16 frames.
Applies MobileNetV2 spatial feature extraction, LSTM recurrent modeling, and temporal
attention on the sliding temporal window to deliver low-latency on-screen activity prediction.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm_attention import get_attention_extraction_model
from src.models.temporal_attention import TemporalAttention
from src.utils.config import load_config

DEFAULT_MODEL_PATHS = {
    "cnn_baseline": Path("outputs/models/cnn_baseline_head.keras"),
    "cnn_lstm": Path("outputs/models/cnn_lstm_head.keras"),
    "cnn_lstm_attention": Path("outputs/models/cnn_lstm_attention_head.keras"),
}


class RealtimeActivityDetector:
    """
    Live streaming activity detector operating on a rolling temporal frame buffer.
    """

    def __init__(
        self,
        model_type: str = "cnn_lstm_attention",
        model_path: Optional[str | Path] = None,
        config_path: str | Path = "configs/default_config.yaml",
        buffer_size: int = 16,
        inference_interval: int = 2,
    ) -> None:
        """
        Initialize real-time detector.

        Args:
            model_type: Architecture ('cnn_lstm_attention', 'cnn_lstm', 'cnn_baseline').
            model_path: Path to model weights .keras.
            config_path: Path to configuration YAML.
            buffer_size: Number of frames in temporal rolling buffer (default 16).
            inference_interval: Perform model inference every N incoming frames for throughput.
        """
        self.config = load_config(config_path)
        self.model_type = model_type.lower()
        self.buffer_size = buffer_size
        self.inference_interval = max(1, inference_interval)

        if model_path is None:
            self.model_path = DEFAULT_MODEL_PATHS.get(
                self.model_type, Path("outputs/models/cnn_lstm_attention_head.keras")
            )
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights missing: {self.model_path.resolve()}")

        # 1. Feature Extractor
        target_size = tuple(self.config["preprocessing"]["target_size"])
        self.target_size = target_size
        self.feature_extractor = CNNFeatureExtractor(
            backbone_name=self.config["model"]["backbone"],
            input_shape=(*target_size, 3),
            trainable=False,
            sequence_length=self.buffer_size,
        )

        # 2. Head Model
        self.head_model = tf.keras.models.load_model(self.model_path, compile=False)
        output_dim = self.head_model.output_shape[-1]
        self.classes = self._resolve_classes(output_dim)

        if self.model_type == "cnn_lstm_attention":
            self.attention_extractor = get_attention_extraction_model(self.head_model)
        else:
            self.attention_extractor = None

        # 3. Rolling temporal buffer
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self.frame_counter = 0

        # State tracking
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.fps_tracker: Deque[float] = deque(maxlen=30)
        self.last_timestamp = time.time()

    def _resolve_classes(self, output_dim: int) -> List[str]:
        """Resolve class names matching the loaded head model's output dimension."""
        cfg_classes = self.config.get("dataset", {}).get("selected_classes")
        if isinstance(cfg_classes, list) and len(cfg_classes) == output_dim:
            return sorted(cfg_classes)

        if output_dim == 5:
            return ["ApplyEyeMakeup", "Basketball", "Biking", "JumpingJack", "WalkingWithDog"]

        class_to_idx_path = PROJECT_ROOT / "outputs" / "class_to_idx.json"
        if class_to_idx_path.exists():
            try:
                with open(class_to_idx_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                sorted_classes = [k for k, _ in sorted(mapping.items(), key=lambda item: item[1])]
                if len(sorted_classes) == output_dim:
                    return sorted_classes
            except Exception:
                pass

        raw_dir = Path(self.config.get("dataset", {}).get("raw_dir", "D:/datasets/UCF-101"))
        if raw_dir.exists():
            dirs = [d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(dirs) == output_dim:
                return sorted(dirs)

        return [f"Class_{i}" for i in range(output_dim)]

    def preprocess_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Resize and normalize BGR frame to (224, 224, 3) float32 in [-1, 1]."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return normalized

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        draw_hud: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Ingest a single frame, update the buffer, execute inference if scheduled,
        and render on-screen HUD graphics.

        Args:
            bgr_frame: Raw camera frame (H, W, 3) BGR format.
            draw_hud: Whether to draw HUD overlay on return frame.

        Returns:
            Tuple[np.ndarray, Optional[Dict[str, Any]]]: (annotated_frame, prediction_dict)
        """
        curr_time = time.time()
        dt = curr_time - self.last_timestamp
        self.last_timestamp = curr_time
        if dt > 0:
            self.fps_tracker.append(1.0 / dt)
        fps = float(np.mean(self.fps_tracker)) if self.fps_tracker else 0.0

        # Preprocess and append to buffer
        norm_frame = self.preprocess_frame(bgr_frame)
        self.frame_buffer.append(norm_frame)
        self.frame_counter += 1

        # Execute inference if buffer is full and interval is reached
        if len(self.frame_buffer) == self.buffer_size and (
            self.frame_counter % self.inference_interval == 0 or self.last_prediction is None
        ):
            frames_tensor = np.array(self.frame_buffer, dtype=np.float32)  # (16, 224, 224, 3)
            feat_seq = self.feature_extractor.extract_sequence_features(frames_tensor)
            feat_batch = np.expand_dims(feat_seq, axis=0)  # (1, 16, 1280)

            attn_weights = None
            if self.model_type == "cnn_lstm_attention" and self.attention_extractor is not None:
                probs_arr, weights_arr = self.attention_extractor.predict(feat_batch, verbose=0)
                probs = probs_arr[0]
                attn_weights = np.squeeze(weights_arr[0]).tolist()
            else:
                probs = self.head_model.predict(feat_batch, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            pred_class = self.classes[pred_idx]
            confidence = float(probs[pred_idx])

            self.last_prediction = {
                "predicted_class": pred_class,
                "confidence": round(confidence, 4),
                "confidence_pct": round(confidence * 100, 1),
                "class_probabilities": {
                    cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(self.classes)
                },
                "attention_weights": attn_weights,
                "fps": round(fps, 1),
                "buffer_ready": True,
            }

        annotated_frame = bgr_frame.copy()
        if draw_hud:
            annotated_frame = self._render_hud(annotated_frame, fps)

        return annotated_frame, self.last_prediction

    def _render_hud(self, frame: np.ndarray, current_fps: float) -> np.ndarray:
        """Render informative HUD banner, confidence meter, and status info on frame."""
        h, w = frame.shape[:2]

        # Top overlay banner (height 90px)
        banner_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Bottom attention bar background (height 45px)
        bottom_h = 45
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - bottom_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bottom, 0.85, frame, 0.15, 0, frame)

        # Buffer status text
        if len(self.frame_buffer) < self.buffer_size:
            status_text = f"BUFFERING TEMPORAL WINDOW: {len(self.frame_buffer)}/{self.buffer_size} FRAMES"
            cv2.putText(
                frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2
            )
        elif self.last_prediction:
            pred_class = self.last_prediction["predicted_class"].upper()
            conf_pct = self.last_prediction["confidence_pct"]

            # Color coding: Green if confidence > 80%, Orange otherwise
            box_color = (0, 220, 100) if conf_pct > 80.0 else (0, 180, 255)

            # Row 1: Action Name (Left) and Confidence % (Right)
            main_text = f"ACTION: {pred_class}"
            cv2.putText(
                frame, main_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
            )

            conf_text = f"{conf_pct:.1f}%"
            cv2.putText(
                frame, conf_text, (w - 110, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.80, box_color, 2
            )

            # Row 2: Confidence Bar + Model / FPS
            bar_x = 18
            bar_w = int(w * 0.45)
            bar_y = 48
            bar_h = 12

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
            fill_w = int((conf_pct / 100.0) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), box_color, -1)

            info_text = f"{self.model_type} | {current_fps:.1f} FPS"
            cv2.putText(
                frame, info_text, (bar_x + bar_w + 15, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1
            )

            # Row 3: Class probabilities summary line
            class_abbr = {
                "ApplyEyeMakeup": "Makeup",
                "Basketball": "Basket",
                "Biking": "Biking",
                "JumpingJack": "Jacks",
                "WalkingWithDog": "Dog",
            }
            prob_items = []
            for cls_name, p_val in self.last_prediction["class_probabilities"].items():
                short_name = class_abbr.get(cls_name, cls_name[:6])
                prob_items.append(f"{short_name}: {p_val*100:.0f}%")
            prob_summary = " | ".join(prob_items)
            cv2.putText(
                frame, prob_summary, (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1
            )

            # Bottom: Attention Sparkline along bottom
            attn_weights = self.last_prediction.get("attention_weights")
            if attn_weights:
                step_w = w / len(attn_weights)
                max_w = max(attn_weights) if max(attn_weights) > 0 else 1.0
                top_thresh = np.percentile(attn_weights, 75)
                for idx, weight in enumerate(attn_weights):
                    bar_height = int((weight / max_w) * (bottom_h - 18))
                    bx1 = int(idx * step_w + 3)
                    bx2 = int((idx + 1) * step_w - 3)
                    by1 = h - bar_height - 6
                    by2 = h - 6
                    col = (0, 140, 255) if weight >= top_thresh else (160, 90, 200)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, -1)

        # Bottom mode label
        mode_label = "TEMPORAL ATTENTION WEIGHTS (16 FRAMES) | PRESS 'q' TO QUIT"
        cv2.putText(
            frame, mode_label, (18, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1
        )

        return frame

    def run_stream(
        self,
        video_source: int | str = 0,
        window_title: str = "Real-Time Early Activity Prediction",
        display: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Open video stream (webcam or file) and execute real-time detection loop.

        Args:
            video_source: 0 for default webcam, or path to video file.
            window_title: OpenCV display window title.
            display: Whether to open interactive GUI window.
            max_frames: Optional limit on frames to process (useful for tests/benchmarks).
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        print(f"\n[STREAM] Running real-time detection on source '{video_source}'...")
        print("[STREAM] Press 'q' or ESC in the display window to stop.\n")

        processed = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed += 1
                annotated_frame, pred = self.process_frame(frame, draw_hud=display)

                if display:
                    cv2.imshow(window_title, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:  # 'q' or ESC
                        print("[STREAM] User requested stop.")
                        break

                if max_frames and processed >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming Human Activity Detection from Camera or Video."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (e.g. 0 for default webcam).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for simulated real-time stream.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lstm_attention",
        choices=["cnn_lstm_attention", "cnn_lstm", "cnn_baseline"],
        help="Architecture to use for real-time inference.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional custom path to .keras model weights.",
    )

    args = parser.parse_args()
    source = args.camera if args.camera is not None else (args.video or 0)

    detector = RealtimeActivityDetector(
        model_type=args.model,
        model_path=args.weights,
    )
    detector.run_stream(video_source=source, display=True)


if __name__ == "__main__":
    main()
