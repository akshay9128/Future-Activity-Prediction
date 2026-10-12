# Future Human Activity Prediction Using CNN + LSTM + Temporal Attention

An end-to-end, beginner-friendly Deep Learning project for video action understanding on the **UCF101** dataset using computer vision, recurrent neural networks, and temporal attention.

---

## 1. Project Overview

Predicting human activities from video sequences is a core challenge in computer vision and artificial intelligence with real-world applications in robotics, autonomous surveillance, sports analytics, and smart healthcare.

This project implements a multi-stage deep learning pipeline:
1. **CNN Spatial Feature Extractor (MobileNetV2):** Encodes individual video frames into compact, rich spatial visual embeddings.
2. **Long Short-Term Memory (LSTM):** Captures temporal dependencies across ordered frame sequences.
3. **Temporal Attention Mechanism:** Dynamically weights the relative importance of distinct time steps (distinguishing key action frames from background motion).

---

## 2. Important Research Definition & Limitation

In video action recognition literature, precision in definitions is critical:

- **Early Action Prediction (Implemented Baseline):** Observing only the initial segment of an ongoing activity (e.g., the first 50% of a video clip) to predict the action class before the action fully concludes.
- **Future Activity Prediction (Anticipation):** Observing an activity $A$ (e.g., *chopping vegetables*) and predicting a completely distinct subsequent activity $B$ (e.g., *frying in a pan*).

> **UCF101 Dataset Limitation:** UCF101 contains video-level action labels rather than sequential transitions across distinct multi-stage activities. Therefore, our primary experimental protocol implements **Early Action Prediction** (predicting the action from early temporal observations). If multi-step future activity prediction is desired, dense transition-labeled datasets (such as Breakfast Actions, EPIC-KITCHENS, or synthetic multi-action concatenations) would be required.

---

## 3. Selected Action Classes (UCF101 Subset)

To ensure rapid experimentation and practical resource utilization on a local computer, the initial focus is on a balanced 5-class subset:
1. `Basketball` (134 videos)
2. `Biking` (134 videos)
3. `WalkingWithDog` (123 videos)
4. `JumpingJack` (123 videos)
5. `ApplyEyeMakeup` (145 videos)

Total videos in active subset: **659 videos**.

---

## 4. Project Directory Structure

```text
Activity-prediction/
├── configs/
│   └── default_config.yaml     # Centralized hyperparameters & paths
├── src/
│   ├── __init__.py
│   ├── data/                   # Video loading, uniform frame sampling, manifests
│   │   ├── __init__.py
│   │   └── dataset_inspector.py
│   ├── models/                 # CNN backbone, LSTM, and Temporal Attention layers
│   │   └── __init__.py
│   └── utils/                  # Random seeding, metrics, path helpers
│       ├── __init__.py
│       └── config.py
├── tests/                      # pytest test suite
│   ├── __init__.py
│   ├── test_environment.py     # Environment, imports, and sanity tests
│   └── test_dataset_inspection.py # Dataset verification tests
├── outputs/                    # Logs, evaluation plots, and saved weights (gitignored)
│   ├── dataset_summary.json    # Phase 2 inspection output
│   ├── checkpoints/
│   └── logs/
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes large binaries, venvs, and datasets
└── README.md                   # Project documentation
```

---

## 5. Technology Stack

- **Language:** Python 3.11
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Computer Vision:** OpenCV (`cv2`)
- **Scientific Computing & Data:** NumPy, Pandas
- **Evaluation & Visualization:** scikit-learn, Matplotlib, Seaborn
- **Testing:** pytest
- **Hardware Acceleration:** NVIDIA CUDA / cuDNN (NVIDIA GeForce RTX 3060 Laptop GPU)

---

## 6. Setup & Getting Started

### Prerequisites
- Windows 10/11
- Python 3.11 installed (`py -3.11`)
- Visual Studio Code

### Virtual Environment Setup
Open PowerShell in the project directory (`D:\project\Activity-prediction`):

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 7. How to Inspect the Dataset (Phase 2)

Run the automated inspection script:
```powershell
python src\data\dataset_inspector.py
```

Run all unit tests:
```powershell
pytest tests -v
```

---

## 8. Development Roadmap

- [x] **Phase 1: Project inspection and clean setup**
- [x] **Phase 2: Dataset verification and class selection** *(Current)*
- [ ] **Phase 3: Data preparation and temporal sampling**
- [ ] **Phase 4: Build a CNN feature extractor (MobileNetV2)**
- [ ] **Phase 5: Create a CNN-only baseline**
- [ ] **Phase 6: Build the CNN + LSTM model**
- [ ] **Phase 7: Add temporal attention mechanism**
- [ ] **Phase 8: Training pipeline & configuration**
- [ ] **Phase 9: Final evaluation & error analysis**
- [ ] **Phase 10: Inference on unseen videos**
- [ ] **Phase 11: Optional interactive demonstration**
- [ ] **Phase 12: Documentation & final verification**
