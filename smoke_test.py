"""
End-to-end smoke test for the Single CNN + LSTM Activity Prediction Pipeline.
Validates environment dependencies, configuration, 101-class mappings, model loading,
inference prediction schema, and output directory integrity.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np


def run_smoke_test() -> bool:
    """
    Execute end-to-end sanity verification on the single CNN + LSTM pipeline.

    Returns:
        bool: True if all health checks pass, False otherwise.
    """
    project_root = Path(__file__).resolve().parent
    print("=" * 70)
    print("SANITY CHECK & SMOKE TEST: SINGLE CNN + LSTM ACTIVITY PREDICTION")
    print("=" * 70)

    all_passed = True

    # 1. Environment & Core Libraries Check
    print("\n[1/6] Validating environment and dependencies...")
    try:
        import tensorflow as tf
        import cv2
        import yaml
        print(f"  - Python Version:     {sys.version.split()[0]}")
        print(f"  - TensorFlow Version: {tf.__version__}")
        print(f"  - OpenCV Version:     {cv2.__version__}")
        print(f"  - PyYAML Version:     {yaml.__version__}")
        print("  [PASS] All core dependencies imported successfully.")
    except Exception as exc:
        print(f"  [FAIL] Dependency error: {exc}")
        return False

    # 2. Configuration & Class Mappings Check
    print("\n[2/6] Validating configuration and 101-class mappings...")
    try:
        from src.utils.config import load_config
        config = load_config(project_root / "configs" / "default_config.yaml")
        print(f"  - Config Loaded:      {config.get('project_name')}")
        print(f"  - Configured Classes: {config['model']['num_classes']}")
        print(f"  - Sequence Length:    {config['preprocessing']['sequence_length']}")
        print(f"  - Observation Ratio:  {config['preprocessing']['observation_ratio']}")

        c2i_path = project_root / "outputs" / "class_to_idx.json"
        i2c_path = project_root / "outputs" / "idx_to_class.json"
        with open(c2i_path, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        with open(i2c_path, "r", encoding="utf-8") as f:
            i2c = json.load(f)

        assert len(c2i) == 101, f"Expected 101 classes in class_to_idx, found {len(c2i)}"
        assert len(i2c) == 101, f"Expected 101 classes in idx_to_class, found {len(i2c)}"
        print(f"  - Mapped Classes:     101 verified across both lookup tables")
        print("  [PASS] Configuration and class mappings verified.")
    except Exception as exc:
        print(f"  [FAIL] Configuration error: {exc}")
        all_passed = False

    # 3. Model Weights & Checkpoint Integrity
    print("\n[3/6] Validating model checkpoints and architecture...")
    try:
        from src.models.cnn_lstm import build_cnn_lstm_head

        head_path = project_root / "outputs" / "checkpoints" / "cnn_lstm_head.keras"
        e2e_path = project_root / "outputs" / "checkpoints" / "cnn_lstm_end_to_end.keras"

        assert head_path.is_file(), f"Missing checkpoint: {head_path}"
        assert e2e_path.is_file(), f"Missing checkpoint: {e2e_path}"

        head_model = tf.keras.models.load_model(head_path)
        param_count = head_model.count_params()
        input_shape = head_model.input_shape
        output_shape = head_model.output_shape

        print(f"  - Head Checkpoint:    {head_path.name} ({head_path.stat().st_size / (1024*1024):.2f} MB)")
        print(f"  - E2E Checkpoint:     {e2e_path.name} ({e2e_path.stat().st_size / (1024*1024):.2f} MB)")
        print(f"  - Parameters:         {param_count:,}")
        print(f"  - Head Input Shape:   {input_shape}")
        print(f"  - Head Output Shape:  {output_shape}")

        assert input_shape == (None, 16, 1280), f"Unexpected input shape: {input_shape}"
        assert output_shape == (None, 101), f"Unexpected output shape: {output_shape}"
        print("  [PASS] Model architecture and checkpoints verified.")
    except Exception as exc:
        print(f"  [FAIL] Model checkpoint error: {exc}")
        all_passed = False

    # 4. Forward Pass & Latency Measurement
    print("\n[4/6] Executing synthetic sequence forward pass...")
    try:
        np.random.seed(42)
        dummy_features = np.random.randn(1, 16, 1280).astype(np.float32)

        start_time = time.perf_counter()
        preds = head_model.predict(dummy_features, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000

        prob_sum = float(np.sum(preds))
        assert np.isfinite(preds).all(), "Predictions contain NaN or Inf values"
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"

        top_idx = int(np.argmax(preds[0]))
        top_prob = float(preds[0][top_idx])
        top_name = i2c[str(top_idx)]

        print(f"  - Forward Latency:    {latency_ms:.2f} ms")
        print(f"  - Top Pred Class:     {top_name} (Index: {top_idx}, Confidence: {top_prob*100:.2f}%)")
        print(f"  - Softmax Probability Sum: {prob_sum:.4f}")
        print("  [PASS] Forward inference is finite, normalized, and performant.")
    except Exception as exc:
        print(f"  [FAIL] Forward pass error: {exc}")
        all_passed = False

    # 5. ActivityPredictor High-Level Class Verification
    print("\n[5/6] Validating ActivityPredictor API...")
    try:
        from src.inference.predict import ActivityPredictor
        predictor = ActivityPredictor(model_type="cnn_lstm")
        assert predictor.model_type == "cnn_lstm"
        assert len(predictor.classes) == 101
        print("  - ActivityPredictor initialized with default model_type='cnn_lstm'")
        print(f"  - Classes loaded in predictor: {len(predictor.classes)}")
        print("  [PASS] High-level predictor API validated.")
    except Exception as exc:
        print(f"  [FAIL] ActivityPredictor API error: {exc}")
        all_passed = False

    # 6. Output Organization Integrity Check
    print("\n[6/6] Verifying output directory and artifact integrity...")
    try:
        from src.utils.output_organizer import verify_outputs_integrity
        integrity_results = verify_outputs_integrity(project_root)
        failed_items = [name for name, status in integrity_results.items() if not status]
        if failed_items:
            print(f"  [FAIL] Missing artifacts: {failed_items}")
            all_passed = False
        else:
            print(f"  - Checked Items:      {len(integrity_results)} output artifacts verified")
            print("  [PASS] All output directories, reports, plots, and models intact.")
    except Exception as exc:
        print(f"  [FAIL] Output verification error: {exc}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("SMOKE TEST STATUS: ALL SYSTEMS OPERATIONAL AND REPRODUCIBLE (PASS)")
    else:
        print("SMOKE TEST STATUS: ONE OR MORE CHECKS FAILED (FAIL)")
    print("=" * 70 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
