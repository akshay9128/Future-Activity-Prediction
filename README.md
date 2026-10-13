# Future Human Activity Prediction Using Single CNN + LSTM

An end-to-end, beginner-friendly Deep Learning project for early human activity prediction across all **101 classes** of the **UCF-101** dataset using a single unified **CNN + LSTM** neural network architecture.

---

## 1. Project Overview

Predicting human activities from video sequences before they fully unfold is a core challenge in artificial intelligence, with applications in robotics, autonomous surveillance, sports analytics, and assisted healthcare.

This project implements an early action prediction pipeline powered by a single **CNN + LSTM** model:
1. **Spatial Visual Feature Extraction:** Uses a frozen **MobileNetV2** backbone pretrained on ImageNet to convert individual video frames into rich 1,280-dimensional feature representations.
2. **Temporal Sequence Modeling:** Uses a single **Long Short-Term Memory (LSTM)** layer (64 units) with recurrent dropout to model action dynamics over time.
3. **Classification Head:** A dense softmax classifier predicting the probability distribution across all **101 UCF-101 activity classes**.

---

## 2. Task Formulation: Early Video Activity Classification

In video action recognition literature:
- **Early Video Activity Classification (Implemented):** Observing only the initial segment of an ongoing activity (the first 50% of the video duration across 16 uniformly sampled frames) and predicting the action class before the activity finishes.
- **Why this formulation?** UCF-101 contains video-level action annotations (one action label per video). By restricting the observation window to the first 50% of the video, the network must anticipate the activity category using early temporal and visual cues.

```text
Full Video Timeline:
[======================= 100% Total Duration =======================]
[---- First 50% (Observed) ----][---- Remaining 50% (Unobserved) ----]
  │   │   │   ...   │
  └───┴───┴─────────┴──► Sample 16 Frames ──► MobileNetV2 ──► Single LSTM ──► Predicted 101 Classes
```

---

## 3. Model Architecture

The project strictly adheres to a single **CNN + LSTM** model:

| Layer / Component | Description | Output Shape | Parameters |
|---|---|---|---|
| **Input Frames** | 16 RGB video frames ($224 \times 224 \times 3$) | `(Batch, 16, 224, 224, 3)` | 0 |
| **CNN Backbone** | MobileNetV2 (ImageNet weights, frozen) | `(Batch, 16, 1280)` | 2,257,984 (Non-trainable) |
| **Temporal LSTM** | Single LSTM layer (64 hidden units, recurrent dropout 0.3) | `(Batch, 64)` | 344,320 (Trainable) |
| **Dropout** | Regularization layer (dropout rate 0.3) | `(Batch, 64)` | 0 |
| **Dense Head** | Fully connected layer with Softmax activation | `(Batch, 101)` | 6,565 (Trainable) |
| **Total Head Parameters** | | | **355,045** |
| **Model Size** | Head: **4.10 MB**, End-to-End: **13.26 MB** | | |

---

## 4. Dataset: UCF-101 (All 101 Classes)

The pipeline is trained and evaluated on the full **UCF-101 Action Recognition Dataset**:
- **Number of Classes:** **101** diverse categories (e.g., `Basketball`, `Biking`, `Archery`, `PlayingPiano`, `SalsaSpin`, `WalkingWithDog`, etc.).
- **Total Videos:** **13,320** video clips.
- **Disjoint Split Protocol:** Partitioned **70% Train / 15% Validation / 15% Held-Out Test** by video group ID to ensure zero data leakage between splits.
- All 101 classes are present in training, validation, and test splits.

---

## 5. Official Benchmark Results

The single CNN + LSTM model demonstrates strong generalization and low latency:

| Metric | Validation Set (303 Videos) | Held-Out Test Set (303 Videos) |
|---|---|---|
| **Loss** | 1.4067 | **1.4993** |
| **Top-1 Accuracy** | 67.66% | **68.98%** |
| **Top-5 Accuracy** | N/A | **82.51%** |
| **Macro Precision** | 0.7029 | Evaluated across 101 classes |
| **Macro Recall** | 0.6766 | Evaluated across 101 classes |
| **Macro F1-Score** | 0.6578 | Evaluated across 101 classes |
| **Forward Latency** | ~2.8 ms (head pass) | **2.78 ms / video sequence** |
| **Generalization Gap** | — | **-1.32%** (No overfitting) |

---

## 6. Quickstart Guide

### Prerequisites
- Windows 10/11, Linux, or macOS
- Python 3.11+
- Virtual environment tool

### Installation

1. **Clone and enter the repository:**
   ```bash
   cd Activity-prediction
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\activate
   ```
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 1-Minute Sanity Check (Smoke Test)

Run the automated smoke test script to verify dependencies, model checkpoints, 101-class mappings, and forward inference in seconds:

```bash
python smoke_test.py
```

Expected output:
```text
======================================================================
SANITY CHECK & SMOKE TEST: SINGLE CNN + LSTM ACTIVITY PREDICTION
======================================================================
[1/6] Validating environment and dependencies...      [PASS]
[2/6] Validating configuration and 101-class mappings... [PASS]
[3/6] Validating model checkpoints and architecture... [PASS]
[4/6] Executing synthetic sequence forward pass...     [PASS]
[5/6] Validating ActivityPredictor API...              [PASS]
[6/6] Verifying output directory and artifact integrity... [PASS]
======================================================================
SMOKE TEST STATUS: ALL SYSTEMS OPERATIONAL AND REPRODUCIBLE (PASS)
======================================================================
```

---

## 7. Running Inference on Videos

Predict the activity in any unseen video using the trained single CNN + LSTM model:

```bash
# Predict on an unseen test video and save the visual prediction dashboard
python -m src.inference.predict --video "D:/datasets/UCF-101/Basketball/v_Basketball_g01_c01.avi" --model cnn_lstm --save-vis outputs/predictions/basketball_prediction.png
```

### Output Example:
- **Predicted Class:** `Basketball` (97.88% confidence)
- **Top-5 Breakdown:**
  1. `Basketball`: 97.88%
  2. `BalanceBeam`: 0.30%
  3. `PlayingPiano`: 0.22%
  4. `MoppingFloor`: 0.16%
  5. `Fencing`: 0.16%
- **Inference Time:** 2,516 ms (including frame extraction and model prediction)
- **Dashboard Saved:** `outputs/predictions/basketball_prediction.png`

---

## 8. Interactive Web Demo (Streamlit)

Launch the full interactive browser interface:

```bash
streamlit run streamlit_app.py
```

### Features:
- **Interactive Video Player:** Upload any video or select sample videos from the UCF-101 held-out test split.
- **Top-5 Probabilities:** Live horizontal bar chart of top model predictions.
- **Sampled Frame Strip:** Visual preview of the 16 uniformly sampled frames from the first 50% observation window.
- **Latency Tracker:** Real-time inference time displayed in milliseconds.

---

## 9. End-to-End Reproduction Pipeline

To reproduce all cached features, model training, and evaluations from scratch, follow these commands in order:

```bash
# 1. Inspect dataset structure and verify 101 classes
python -m src.data.inspect_dataset

# 2. Generate disjoint 70/15/15 manifest and cache MobileNetV2 features
python -m src.data.dataset_preparation

# 3. Train the single CNN + LSTM model (15 epochs, batch size 16)
python -m src.models.train --model cnn_lstm --epochs 15 --batch-size 16

# 4. Evaluate on 303 held-out test videos
python -m src.evaluation.evaluate_models

# 5. Organize output directories and compile evaluation reports
python -c "from src.utils.output_organizer import organize_and_save_artifacts; organize_and_save_artifacts()"
```

For complete environment specifications and exact configurations, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 10. Repository Directory Structure

```text
Activity-prediction/
├── configs/
│   └── default_config.yaml         # Centralized hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── dataset_inspector.py    # Class discovery and video integrity auditing
│   │   └── dataset_preparation.py  # 70/15/15 disjoint splitting & feature caching
│   ├── models/
│   │   ├── cnn_feature_extractor.py # Frozen MobileNetV2 frame encoder
│   │   ├── cnn_lstm.py             # Single CNN + LSTM model architecture
│   │   └── train.py                # Model training and validation script
│   ├── evaluation/
│   │   └── evaluate_models.py      # Held-out test evaluation & confusion matrix generator
│   ├── inference/
│   │   └── predict.py              # CLI and programmatic video predictor
│   └── utils/
│       ├── config.py               # YAML configuration loader
│       └── output_organizer.py     # Artifact organization and integrity validation
├── tests/                          # 86 automated unit and integration tests
│   ├── test_smoke_pipeline.py      # Automated smoke test & determinism verification
│   ├── test_output_organization.py # Output directory and artifact checks
│   ├── test_cnn_lstm.py            # CNN + LSTM unit tests
│   ├── test_inference.py           # Inference API and schema tests
│   └── test_environment.py         # Environment dependency tests
├── outputs/
│   ├── checkpoints/                # Production model weights (cnn_lstm_head.keras)
│   ├── models/                     # Model weights export
│   ├── plots/                      # High-res curves, confusion matrices, prediction charts
│   ├── reports/                    # JSON and Markdown evaluation reports
│   ├── predictions/                # Inference results and visual dashboards
│   ├── cache/                      # Precomputed feature arrays (.npz)
│   ├── class_to_idx.json           # 101-class mapping lookup
│   ├── idx_to_class.json           # Inverse class lookup
│   └── dataset_manifest.csv        # 13,320 video metadata manifest
├── smoke_test.py                   # One-command standalone health check script
├── streamlit_app.py                # Interactive web demo application
├── requirements.txt                # Pinned Python package dependencies
├── REPRODUCIBILITY.md              # Deterministic reproducibility guide
├── .gitignore                      # Excludes large binaries, weights, and caches
└── README.md                       # Main project documentation
```

---

## 11. Automated Testing

Run the comprehensive pytest test suite:

```bash
pytest -v
```

**Status:** **86 passed, 0 failed** (100% pass rate).

Test coverage includes:
- Python 3.9+ environment and dependency checks
- Disjoint 70/15/15 splitting and reproducibility checks
- 101-class representation across all splits
- Model output tensor shapes, gradient flow, and parameter hierarchies
- Numerical determinism and absence of `NaN`/`Inf` weights
- Top-5 prediction schema, latency tracking, and error handling for missing checkpoints
- Output directory integrity and artifact preservation

---

## 12. License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
