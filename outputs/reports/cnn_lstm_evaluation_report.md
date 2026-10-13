# Single CNN + LSTM Model Evaluation Report

**Task Formulation:** Early Video Activity Classification (First 50% Observed)  
**Dataset:** UCF-101 (All 101 Classes)  
**Backbone:** Frozen MobileNetV2 (`ImageNet` weights, 1,280 features per frame)  
**Sequence Modeling:** Single LSTM layer (64 units, recurrent dropout 0.3)  
**Classification Head:** Dense Softmax (101 output logits)  

---

## 1. Executive Performance Summary

| Metric | Validation Set (303 Videos) | Held-Out Test Set (303 Videos) |
|---|---|---|
| **Loss** | 1.4067 | 1.4993 |
| **Top-1 Accuracy** | 67.66% | 68.98% |
| **Top-5 Accuracy** | N/A | 82.51% |
| **Macro Precision** | 0.7029 | Evaluated across 101 classes |
| **Macro Recall** | 0.6766 | Evaluated across 101 classes |
| **Macro F1-Score** | 0.6578 | Evaluated across 101 classes |
| **Inference Latency** | ~2.8 ms (head forward) | 2.78 ms/video |
| **Parameter Count** | 355,045 parameters | 355,045 parameters |
| **Model Size** | 4.10 MB | 4.10 MB |

---

## 2. Key Observations

1. **Strong Generalization:** Test accuracy (68.98%) aligns tightly with validation accuracy (67.66%), demonstrating healthy generalization without severe overfitting.
2. **Top-5 Accuracy:** The model achieves 82.51% Top-5 accuracy across 101 fine-grained sports and human activities.
3. **Temporal Processing:** Recurrent sequence modeling via a single 64-unit LSTM processes the 16 observed frames efficiently with minimal latency overhead (~2.78 ms per video sequence on CPU).

---

## 3. Preserved Artifacts Inventory

- **Checkpoints:** `outputs/checkpoints/cnn_lstm_head.keras`, `outputs/checkpoints/cnn_lstm_end_to_end.keras`
- **Plots & Curves:** `outputs/plots/cnn_lstm_training_curves.png`, `outputs/plots/cnn_lstm_test_confusion_matrix.png`
- **Prediction Outputs:** `outputs/predictions/basketball_lstm_prediction.json`, `outputs/predictions/biking_lstm_prediction.json`
- **Class Mappings:** `outputs/class_to_idx.json`, `outputs/idx_to_class.json`
