"""
CNN + LSTM + Temporal Attention Training and Validation Pipeline.

Trains the recurrent temporal attention network on extracted frame features,
evaluates on validation set, computes confusion matrix and classification reports,
visualizes attention weight distributions across time steps, and serializes both
head and end-to-end models.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manifest import load_manifest
from src.data.feature_cache import get_or_extract_split_features
from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm_attention import (
    build_cnn_lstm_attention_head,
    build_end_to_end_cnn_lstm_attention,
    get_attention_extraction_model,
)
from src.utils.config import load_config


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Plot and save loss and accuracy curves for training and validation."""
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history.history["loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history.history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("CNN + LSTM + Attention - Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], "r--s", label="Validation Accuracy")
    ax2.set_title("CNN + LSTM + Attention - Categorical Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training curves to: {output_path.resolve()}", flush=True)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], output_path: Path
) -> None:
    """Plot and save normalized confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="CNN + LSTM + Attention Validation Confusion Matrix",
        ylabel="Ground Truth Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved confusion matrix to: {output_path.resolve()}", flush=True)


def plot_attention_weights_by_class(
    extractor_model: tf.keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Plot temporal attention distributions across the 16 observed time steps
    for representative correctly-classified validation samples of each class.
    """
    probs, weights = extractor_model.predict(x_val, verbose=0)
    preds = np.argmax(probs, axis=1)
    weights = np.squeeze(weights, axis=-1)  # (N, 16)

    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 2.2 * len(class_names)), sharex=True)
    if len(class_names) == 1:
        axes = [axes]

    timesteps = np.arange(1, weights.shape[1] + 1)
    # Colors for distinct visual presentation
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        # Find correctly classified samples for this class
        matching_indices = np.where((y_val == idx) & (preds == idx))[0]
        if len(matching_indices) == 0:
            matching_indices = np.where(y_val == idx)[0]

        # Average attention across matching validation samples for class
        class_weights = weights[matching_indices]
        mean_weights = np.mean(class_weights, axis=0)
        std_weights = np.std(class_weights, axis=0)

        color = colors[idx % len(colors)]
        ax.plot(timesteps, mean_weights, marker="o", color=color, linewidth=2, label=f"{class_name} (n={len(matching_indices)})")
        ax.fill_between(timesteps, mean_weights - std_weights, mean_weights + std_weights, color=color, alpha=0.2)
        ax.axhline(1.0 / len(timesteps), color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Uniform baseline")
        ax.set_ylabel("Attention Weight", fontsize=9)
        ax.set_title(f"Class: {class_name}", fontsize=11, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Sampled Timestep (1 to 16, within first 50% of video)", fontsize=10)
    axes[-1].set_xticks(timesteps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved temporal attention visualization to: {output_path.resolve()}", flush=True)


def train_cnn_lstm_attention(config_path: Path | str = "configs/default_config.yaml") -> dict:
    """
    Main training routine for the CNN + LSTM + Temporal Attention model.

    Returns:
        dict: Evaluation metrics summary.
    """
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_seed(seed)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    patience = config["training"]["early_stopping_patience"]

    manifest_path = Path(config["paths"]["manifest_csv"])
    manifest_df = load_manifest(manifest_path)
    selected_classes = sorted(config["dataset"]["selected_classes"])
    num_classes = len(selected_classes)

    lstm_units = config["model"].get("lstm_units", 64)
    dropout_rate = config["model"].get("dropout_rate", 0.3)
    attention_units = lstm_units

    print("==================================================", flush=True)
    print("      Training CNN + LSTM + Attention Model       ", flush=True)
    print("==================================================", flush=True)
    print(f"LSTM Units     : {lstm_units}", flush=True)
    print(f"Attention Units: {attention_units}", flush=True)
    print(f"Dropout Rate   : {dropout_rate}", flush=True)
    print(f"Epochs         : {epochs}", flush=True)
    print(f"Batch Size     : {batch_size}", flush=True)
    print(f"Learning Rate  : {lr}", flush=True)
    print(f"Classes ({num_classes})   : {', '.join(selected_classes)}", flush=True)
    print("--------------------------------------------------", flush=True)

    # 1. Feature Extractor
    seq_len = config["preprocessing"]["sequence_length"]
    target_size = tuple(config["preprocessing"]["target_size"])
    obs_ratio = config["preprocessing"]["observation_ratio"]

    extractor = CNNFeatureExtractor(
        backbone_name=config["model"]["backbone"],
        input_shape=(*target_size, 3),
        trainable=False,
        sequence_length=seq_len,
    )

    # 2. Extract or load cached features
    cache_dir = Path("outputs/cache")
    x_train, y_train = get_or_extract_split_features(
        manifest_df, "train", extractor, cache_dir, seq_len, target_size, obs_ratio
    )
    x_val, y_val = get_or_extract_split_features(
        manifest_df, "val", extractor, cache_dir, seq_len, target_size, obs_ratio
    )

    print(f"\nTraining Data   : Features {x_train.shape}, Labels {y_train.shape}", flush=True)
    print(f"Validation Data : Features {x_val.shape}, Labels {y_val.shape}", flush=True)

    # 3. Build Model Head
    head_model = build_cnn_lstm_attention_head(
        sequence_length=seq_len,
        feature_dim=extractor.feature_dim,
        lstm_units=lstm_units,
        attention_units=attention_units,
        dense_units=lstm_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    head_model.summary()

    # 4. Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print("\nStarting CNN + LSTM + Attention Training...", flush=True)
    history = head_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, lr_reducer],
        verbose=1,
    )

    # 5. Evaluate on Validation Set
    val_loss, val_acc = head_model.evaluate(x_val, y_val, verbose=0)
    val_probs = head_model.predict(x_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    cm = confusion_matrix(y_val, val_preds)
    report = classification_report(
        y_val, val_preds, target_names=selected_classes, output_dict=True
    )

    print("\n==================================================", flush=True)
    print(f"Validation Results: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f} ({val_acc*100:.2f}%)", flush=True)
    print("==================================================", flush=True)
    print(classification_report(y_val, val_preds, target_names=selected_classes), flush=True)

    # 6. Save Plots & Visualizations
    plot_training_curves(history, Path("outputs/logs/cnn_lstm_attention_training.png"))
    plot_confusion_matrix(
        cm, selected_classes, Path("outputs/logs/cnn_lstm_attention_confusion_matrix.png")
    )

    # Attention Weights Visualization
    attn_extractor = get_attention_extraction_model(head_model)
    plot_attention_weights_by_class(
        attn_extractor,
        x_val,
        y_val,
        selected_classes,
        Path("outputs/logs/temporal_attention_weights_visualization.png"),
    )

    # 7. Save Models
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    head_save_path = models_dir / "cnn_lstm_attention_head.keras"
    head_model.save(head_save_path)
    print(f"[SAVED] Saved attention head model to: {head_save_path.resolve()}", flush=True)

    # End-to-end model (takes raw video frames directly)
    e2e_model = build_end_to_end_cnn_lstm_attention(
        cnn_frame_model=extractor.frame_model,
        attention_head_model=head_model,
        sequence_length=seq_len,
        input_shape=(*target_size, 3),
    )
    e2e_save_path = models_dir / "cnn_lstm_attention_end_to_end.keras"
    e2e_model.save(e2e_save_path)
    print(f"[SAVED] Saved end-to-end attention model to: {e2e_save_path.resolve()}", flush=True)

    # 8. Save Metrics
    metrics_summary = {
        "model_name": "CNN_LSTM_Attention",
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_samples_total": len(y_val),
        "classes": selected_classes,
        "classification_report": report,
        "epochs_trained": len(history.history["loss"]),
    }
    metrics_path = Path("outputs/logs/cnn_lstm_attention_val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[SAVED] Saved validation metrics to: {metrics_path.resolve()}", flush=True)

    return metrics_summary


if __name__ == "__main__":
    train_cnn_lstm_attention()
