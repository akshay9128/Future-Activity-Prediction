# Reproducibility Guide & Environment Specifications

This document provides complete, deterministic instructions for reproducing all results, benchmarks, checkpoints, and evaluation artifacts for the **Future Human Activity Prediction** project using the **Single CNN + LSTM** model on the **UCF-101** dataset.

---

## 1. Task Definition & Formulation

- **Task:** Early Video Activity Classification
- **Observation Window:** Initial **50%** of total video duration (`observation_ratio: 0.5`)
- **Sampling Scheme:** **16** uniformly sampled RGB frames from the observed window
- **Target:** Classify the ongoing activity into one of **101** fine-grained action classes
- **Dataset:** UCF-101 Action Recognition Dataset (13,320 videos across 101 action classes)

---

## 2. Hardware & Software Specifications

| Component | Specification |
|---|---|
| **Operating System** | Windows 10/11 (PowerShell / Command Prompt) |
| **Python Runtime** | Python 3.11.0 (64-bit) |
| **Deep Learning Framework** | TensorFlow 2.17.1 (Keras 3 / tf.keras) |
| **Computer Vision** | OpenCV (`opencv-python` 4.11.0) |
| **Numerical Processing** | NumPy 1.26.4 |
| **Data Handling** | Pandas 2.2.3 |
| **Visualization & Evaluation** | Matplotlib 3.10.0, Seaborn 0.13.2, Scikit-learn 1.6.1 |
| **Web Demo Framework** | Streamlit 1.42.0 |
| **Hardware** | Intel/AMD CPU with oneDNN optimizations enabled (AVX2 / AVX512) |
| **Global Random Seed** | `42` (configured across NumPy, Python random, and TensorFlow) |

---

## 3. Single CNN + LSTM Model Architecture

The project strictly employs a single **CNN + LSTM** model:

1. **Visual Backbone (Spatial Feature Extractor):**
   - Architecture: MobileNetV2 pretrained on ImageNet
   - Weights: Frozen (`trainable = False`)
   - Input: `(Batch, 16, 224, 224, 3)` normalized using MobileNetV2 preprocessing mapping to $[-1, 1]$
   - Output Feature Representation: `(Batch, 16, 1280)`
2. **Temporal Sequence Modeling:**
   - Layer: Single LSTM layer
   - Hidden Units: `64`
   - Recurrent Dropout: `0.3`
   - Return Sequences: `False` (aggregates sequence into a single 64-dimensional temporal state vector)
3. **Classification Head:**
   - Dense layer with `101` output units
   - Activation: `Softmax`
   - Total Parameters: **355,045** parameters (all trainable in the head)
   - Checkpoint Weight Size: **4.10 MB** (`cnn_lstm_head.keras`), **13.26 MB** (`cnn_lstm_end_to_end.keras`)

---

## 4. Dataset Splitting & Preprocessing Protocol

To prevent data leakage, splits are partitioned **disjointly by video group ID** (e.g., all clips `g01` belong strictly to one split):
- **Train Set:** 70% of groups (~9,324 videos total; 10 videos per class cached for training = 1,010 videos)
- **Validation Set:** 15% of groups (~1,998 videos total; 3 videos per class cached = 303 videos)
- **Held-Out Test Set:** 15% of groups (~1,998 videos total; 3 videos per class cached = 303 videos)
- **Class Representation:** All 101 classes are represented in train, validation, and test splits.

---

## 5. End-to-End Reproduction Steps

Execute the following sequential terminal commands to reproduce the pipeline from scratch:

### Step 1: Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install exact pinned requirements
pip install -r requirements.txt
```

### Step 2: Dataset Verification & Inspection
```powershell
# Inspect UCF-101 directory and verify 101 classes
python -m src.data.inspect_dataset
```

### Step 3: Dataset Splitting & Feature Caching
```powershell
# Generate disjoint 70/15/15 manifest and precompute MobileNetV2 feature tensors
python -m src.data.dataset_preparation
```
*Outputs: `outputs/dataset_manifest.csv`, `outputs/class_to_idx.json`, `outputs/idx_to_class.json`, `outputs/cache/*.npz`*

### Step 4: Model Training (Single CNN + LSTM)
```powershell
# Train the CNN + LSTM model on cached feature representations
python -m src.models.train --model cnn_lstm --epochs 15 --batch-size 16
```
*Outputs: `outputs/models/cnn_lstm_head.keras`, `outputs/models/cnn_lstm_end_to_end.keras`, `outputs/logs/cnn_lstm_training.png`*

### Step 5: Held-Out Test Evaluation
```powershell
# Evaluate trained model on 303 unseen test videos
python -m src.evaluation.evaluate_models
```
*Outputs: `outputs/logs/test_set_evaluation.json`, `outputs/plots/cnn_lstm_test_confusion_matrix.png`*

### Step 6: Artifact Organization & Verification
```powershell
# Standardize directory structure and verify artifact completeness
python -c "from src.utils.output_organizer import organize_and_save_artifacts; organize_and_save_artifacts()"
```

### Step 7: Sanity & Smoke Test
```powershell
# Run the automated smoke test script
python smoke_test.py
```

### Step 8: Inference on Unseen Test Videos
```powershell
# Predict activity on a held-out test video
python -m src.inference.predict --video "D:/datasets/UCF-101/Basketball/v_Basketball_g01_c01.avi" --model cnn_lstm --save-vis outputs/predictions/basketball_prediction.png
```

### Step 9: Launch Interactive Web Application
```powershell
# Run the Streamlit web dashboard
streamlit run streamlit_app.py
```

---

## 6. Official Benchmark Targets

When reproducing results using random seed `42`, expected benchmark metrics should fall within the specified tolerances:

| Metric | Target Value | Tolerance |
|---|---|---|
| **Validation Loss** | 1.4067 | $\pm 0.05$ |
| **Validation Accuracy** | 67.66% | $\pm 1.5\%$ |
| **Held-Out Test Loss** | 1.4993 | $\pm 0.05$ |
| **Held-Out Test Accuracy** | 68.98% | $\pm 1.5\%$ |
| **Held-Out Top-5 Accuracy** | 82.51% | $\pm 1.5\%$ |
| **Inference Latency (Forward)** | 2.78 ms / video | $\pm 1.0$ ms (CPU dependent) |
| **Total Head Parameters** | 355,045 | Exact |

---

## 7. Limitations & Edge Cases

1. **Closed-Set Assumption:** The model is trained on the 101 classes of the UCF-101 taxonomy. Inputs containing actions outside this taxonomy will be mapped to the closest semantically related class.
2. **Partial Observation Constraint:** The model assumes the video depicts an ongoing action from the initial 50% observation period. Videos where the activity begins exclusively in the second half may exhibit lower confidence.
3. **Fixed Sampling Rate:** Videos shorter than 16 frames are padded by repeating the last valid frame.
4. **Hardware Latency Variance:** End-to-end inference includes OpenCV frame decoding (~2.5s on CPU) and LSTM forward pass (~2.8 ms). Using GPU acceleration for decoding and feature extraction further reduces latency.
