"""
Interactive Web Application for Human Activity Prediction.

Provides a Streamlit graphical interface for early action prediction:
- Supports single-video inference on user-uploaded videos or dataset samples.
- Interactive observation ratio control (testing prediction at 20%, 50%, 80%).
- Model switching between CNN Baseline, CNN + LSTM, and CNN + LSTM + Attention.
- Visual frame strip gallery and learned temporal attention keyframe charts.
- Side-by-side comparative benchmarking across all 3 architectures.
- Real-time streaming detection controls and system telemetry.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manifest import load_manifest
from src.inference.predict import ActivityPredictor
from src.utils.config import load_config


def setup_page() -> None:
    """Configure page layout and custom CSS styling."""
    try:
        st.set_page_config(
            page_title="Activity Prediction AI",
            page_icon="🏃",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .prediction-title {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2ca02c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_cached_predictor(model_type: str) -> ActivityPredictor:
    """Cache ActivityPredictor instances across Streamlit reruns."""
    return ActivityPredictor(model_type=model_type)


@st.cache_data(show_spinner=False)
def ensure_web_compatible_video(video_path: str) -> str:
    """Ensure video is playable in HTML5 web browsers (transcodes AVI/MOV to H.264 MP4 if needed)."""
    in_path = Path(video_path)
    if not in_path.exists():
        return video_path

    # Cache transcoded browser-compatible videos
    cache_dir = Path("outputs/cache/web_videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{in_path.stem}_web.mp4"

    if out_path.exists():
        return str(out_path)

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(in_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "faststart",
            "-an",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return str(out_path)
    except Exception:
        return str(in_path)



@st.cache_data
def get_sample_videos_from_manifest() -> Dict[str, str]:
    """Retrieve verified sample videos from the test manifest for quick demonstration."""
    manifest_path = Path("outputs/dataset_manifest.csv")
    samples = {}
    if manifest_path.exists():
        df = load_manifest(manifest_path)
        test_df = df[df["split"] == "test"]
        for cls_name in sorted(test_df["class_name"].unique()):
            match = test_df[test_df["class_name"] == cls_name].iloc[0]
            samples[f"{cls_name} - {Path(match['video_path']).name}"] = match["video_path"]
    return samples


def render_attention_chart(attention_weights: List[float], top_frames: List[int]) -> None:
    """Render Matplotlib attention weight chart with highlighted keyframes."""
    seq_len = len(attention_weights)
    timesteps = np.arange(1, seq_len + 1)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = ["#d95f02" if t in top_frames else "#7570b3" for t in timesteps]

    ax.bar(timesteps, attention_weights, color=colors, edgecolor="black", width=0.6)
    ax.axhline(
        1.0 / seq_len,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Uniform Baseline ({1.0/seq_len:.4f})",
    )

    ax.set_xticks(timesteps)
    ax.set_xlabel("Sampled Timestep (First 50% of Video)", fontsize=9)
    ax.set_ylabel("Attention Score", fontsize=9)
    ax.set_title("Learned Temporal Attention (Salient Keyframes in Orange)", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def main() -> None:
    """Main Streamlit application dashboard."""
    setup_page()

    st.markdown('<div class="main-header">🏃 Future Human Activity Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Early Action Recognition with Single CNN + LSTM (UCF-101: 101 Classes)</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ Model Configuration")

    model_options = {
        "CNN + LSTM": "cnn_lstm",
    }
    selected_model_label = st.sidebar.selectbox("Active Architecture", list(model_options.keys()))
    selected_model_key = "cnn_lstm"
    st.sidebar.caption("Unified Model: MobileNetV2 (Frozen) + Single LSTM (64 units) + 101 Classes")

    obs_ratio = st.sidebar.slider(
        "Observation Ratio (Early Action Window)",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Fraction of video duration observed from the beginning to predict the activity.",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Input Source")
    input_mode = st.sidebar.radio(
        "Choose Video Input",
        ["Pre-loaded Test Samples", "Upload Video File (.avi, .mp4)"],
    )

    video_path_to_run: Optional[str] = None

    if input_mode == "Pre-loaded Test Samples":
        sample_map = get_sample_videos_from_manifest()
        if sample_map:
            chosen_sample = st.sidebar.selectbox("Select Sample Video", list(sample_map.keys()))
            video_path_to_run = sample_map[chosen_sample]
        else:
            st.sidebar.warning("Manifest not found. Please upload a video file.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload video file", type=["avi", "mp4", "mov", "mkv", "webm"])
        if uploaded_file is not None:
            uploads_dir = Path("outputs/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            saved_upload_path = uploads_dir / f"upload_{uploaded_file.name}"
            with open(saved_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path_to_run = str(saved_upload_path)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🎯 Single Video Prediction",
            "📈 Benchmark Metrics (CNN + LSTM)",
            "📹 Live Real-Time Detector",
        ]
    )

    # ------------------ TAB 1: Single Video Prediction ------------------
    with tab1:
        if not video_path_to_run:
            st.info("ℹ️ Please select a sample video or upload a video file from the sidebar to begin.")
            return

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Input Video Clip")
            st.caption(f"File: `{Path(video_path_to_run).name}` | Observation Window: `{int(obs_ratio*100)}%`")
            playable_video_path = ensure_web_compatible_video(video_path_to_run)
            st.video(playable_video_path)

        with col2:
            st.subheader("Prediction Result (CNN + LSTM)")
            if st.button("🚀 Run Early Activity Prediction", type="primary"):
                with st.spinner("Processing video frames and running deep neural inference..."):
                    predictor = get_cached_predictor("cnn_lstm")
                    t0 = time.time()
                    result = predictor.predict_video(
                        video_path_to_run,
                        observation_ratio=obs_ratio,
                        return_frames=True,
                    )
                    latency_ms = (time.time() - t0) * 1000.0

                    pred_class = result["predicted_class"]
                    conf = result["confidence_pct"]

                    # Display Top Metric
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <span style="font-size:0.9rem; color:#666;">PREDICTED ACTIVITY</span>
                            <div class="prediction-title">{pred_class.upper()}</div>
                            <div style="font-size:1.1rem; color:#333; margin-top:0.3rem;">
                                Confidence: <strong>{conf:.1f}%</strong> | Latency: <strong>{latency_ms:.1f} ms</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("### Top-5 Predicted Activities")
                    top5 = result.get("top_5_predictions")
                    if top5:
                        for entry in top5:
                            r = entry["rank"]
                            c_name = entry["class_name"]
                            c_pct = entry["confidence_pct"]
                            badge = "🥇 " if r == 1 else ("🥈 " if r == 2 else ("🥉 " if r == 3 else f"#{r} "))
                            st.write(f"**{badge}{c_name}**: `{c_pct:.2f}%`")
                            st.progress(float(min(max(c_pct / 100.0, 0.0), 1.0)))
                    else:
                        for cls_name, prob in list(result["class_probabilities"].items())[:5]:
                            st.write(f"**{cls_name}**: `{prob*100:.1f}%`")
                            st.progress(float(prob))

                    with st.expander("View All 101 Class Probabilities"):
                        sorted_probs = sorted(result["class_probabilities"].items(), key=lambda x: x[1], reverse=True)
                        for c_name, p_val in sorted_probs:
                            st.write(f"**{c_name}**: `{p_val*100:.2f}%`")

                    # Frame Strip Gallery
                    if "display_frames" in result:
                        st.markdown("### Sampled Frames (Early Observation Window)")
                        st.caption("16 frames uniformly extracted from the first half of the video duration:")
                        frames = result["display_frames"]

                        cols = st.columns(8)
                        for idx in range(8):
                            cols[idx].image(
                                frames[idx],
                                caption=f"T={idx+1}",
                                use_container_width=True,
                            )
                        cols2 = st.columns(8)
                        for idx in range(8, 16):
                            cols2[idx - 8].image(
                                frames[idx],
                                caption=f"T={idx+1}",
                                use_container_width=True,
                            )

    # ------------------ TAB 2: Benchmark Metrics (Single CNN + LSTM) ------------------
    with tab2:
        st.subheader("📈 Quantitative Evaluation & Benchmarks (Single CNN + LSTM)")
        eval_report_path = Path("outputs/reports/cnn_lstm_evaluation_report.json")

        if eval_report_path.exists():
            with open(eval_report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Validation Accuracy", f"{rep['validation_metrics']['val_accuracy_pct']}%", f"Loss: {rep['validation_metrics']['val_loss']:.4f}")
            col_m2.metric("Held-Out Test Accuracy", f"{rep['held_out_test_metrics']['test_accuracy_pct']}%", f"Loss: {rep['held_out_test_metrics']['test_loss']:.4f}")
            col_m3.metric("Top-5 Accuracy", f"{rep['held_out_test_metrics']['top5_accuracy_pct']}%", "Across 101 classes")
            col_m4.metric("Inference Latency", f"{rep['held_out_test_metrics']['inference_latency_ms']} ms", "355,045 parameters")

            st.markdown("---")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                cm_plot_path = Path("outputs/plots/cnn_lstm_test_confusion_matrix.png")
                if cm_plot_path.exists():
                    st.image(str(cm_plot_path), caption="Held-Out Test Confusion Matrix (101 Classes)", use_container_width=True)
            with col_p2:
                curves_path = Path("outputs/plots/cnn_lstm_training_curves.png")
                if curves_path.exists():
                    st.image(str(curves_path), caption="Training & Validation Loss / Accuracy Curves", use_container_width=True)
        else:
            st.info("Evaluation report not found at outputs/reports/cnn_lstm_evaluation_report.json.")

    # ------------------ TAB 3: Live Real-Time Detector ------------------
    with tab3:
        st.subheader("📹 Live Real-Time Webcam Activity Detection (CNN + LSTM)")
        st.markdown(
            """
            This project includes a dedicated **Live Streaming Engine** featuring a **16-frame rolling FIFO buffer**
            that executes MobileNetV2 + Single LSTM inference at **25–30+ FPS** with an on-screen HUD.
            """
        )

        st.markdown(
            """
            #### How to Run the Real-Time Webcam Stream:
            Open a terminal and run:
            ```powershell
            .\\venv\\Scripts\\python.exe -m src.inference.realtime_detector --camera 0 --model cnn_lstm
            ```
            
            #### Features of Real-Time Detection:
            - **Instant Early Prediction**: Begins classifying actions within the first 16 frames of movement.
            - **Live Heads-Up Display (HUD)**: On-screen action name, confidence gauge, and FPS counter.
            - **Press 'q' or ESC** in the camera window to safely exit.
            """
        )


if __name__ == "__main__":
    main()
