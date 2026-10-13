"""
CNN-Only Baseline Training and Validation Script.

Trains the CNN-only baseline on early observation features, evaluates performance
on the held-out validation set, generates training curves, plots a confusion matrix,
and persists trained model artifacts.
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
from src.models.cnn_baseline import (
    build_cnn_baseline_head,
    build_end_to_end_cnn_baseline,
)
from src.models.cnn_feature_extractor import CNNFeatureExtractor
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
    ax1.set_title("CNN Baseline - Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], "r--s", label="Validation Accuracy")
    ax2.set_title("CNN Baseline - Categorical Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training curves to: {output_path.resolve()}")


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], output_path: Path
) -> None:
    """Plot and save normalized confusion matrix."""
    n_classes = len(class_names)
    figsize = (6, 5) if n_classes <= 10 else (12, 10)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if n_classes <= 15:
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Validation Confusion Matrix",
            ylabel="Ground Truth Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate numbers in cells
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
    else:
        ax.set(
            title=f"Validation Confusion Matrix ({n_classes} Classes)",
            ylabel="Ground Truth Label Index",
            xlabel="Predicted Label Index",
        )
        tick_step = max(1, n_classes // 10)
        ticks = np.arange(0, n_classes, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved confusion matrix to: {output_path.resolve()}")


def train_cnn_baseline(config_path: Path | str = "configs/default_config.yaml") -> dict:
    """
    Main training routine for the CNN baseline.

    Returns:
        dict: Evaluation metrics dictionary.
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
    cfg_classes = config["dataset"].get("selected_classes")
    if isinstance(cfg_classes, list):
        selected_classes = sorted(cfg_classes)
    else:
        selected_classes = sorted(manifest_df["class_name"].unique())
    num_classes = len(selected_classes)

    print("==================================================", flush=True)
    print("            Training CNN-Only Baseline            ", flush=True)
    print("==================================================", flush=True)

    print(f"Epochs         : {epochs}", flush=True)
    print(f"Batch Size     : {batch_size}", flush=True)
    print(f"Learning Rate  : {lr}", flush=True)
    class_str = ", ".join(selected_classes) if num_classes <= 5 else f"{num_classes} classes ({selected_classes[0]} ... {selected_classes[-1]})"
    print(f"Classes ({num_classes})   : {class_str}", flush=True)
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
    max_train = config["dataset"].get("max_train_per_class", None)
    max_val = config["dataset"].get("max_val_per_class", None)
    x_train, y_train = get_or_extract_split_features(
        manifest_df, "train", extractor, cache_dir, seq_len, target_size, obs_ratio, max_samples_per_class=max_train
    )
    x_val, y_val = get_or_extract_split_features(
        manifest_df, "val", extractor, cache_dir, seq_len, target_size, obs_ratio, max_samples_per_class=max_val
    )

    print(f"\nTraining Data   : Features {x_train.shape}, Labels {y_train.shape}")
    print(f"Validation Data : Features {x_val.shape}, Labels {y_val.shape}")

    # 3. Build Model Head
    head_model = build_cnn_baseline_head(
        sequence_length=seq_len,
        feature_dim=extractor.feature_dim,
        dense_units=config["model"].get("lstm_units", 64),
        num_classes=num_classes,
        dropout_rate=config["model"]["dropout_rate"],
    )

    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    head_model.summary()

    # 4. Train Model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting Training...")
    history = head_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    # 5. Evaluate on Validation Set
    val_loss, val_acc = head_model.evaluate(x_val, y_val, verbose=0)
    val_probs = head_model.predict(x_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    class_indices = list(range(len(selected_classes)))
    cm = confusion_matrix(y_val, val_preds, labels=class_indices)
    report = classification_report(
        y_val,
        val_preds,
        labels=class_indices,
        target_names=selected_classes,
        output_dict=True,
        zero_division=0,
    )

    print("\n==================================================")
    print(f"Validation Results: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f} ({val_acc*100:.2f}%)")
    print("==================================================")
    if num_classes <= 10:
        print(classification_report(y_val, val_preds, labels=class_indices, target_names=selected_classes, zero_division=0))
    else:
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        print(f"Validation Macro F1: {macro_f1:.4f} across {num_classes} classes")

    # 6. Save Plots & Logs
    plot_training_curves(history, Path("outputs/logs/cnn_baseline_training.png"))
    plot_confusion_matrix(
        cm, selected_classes, Path("outputs/logs/cnn_baseline_confusion_matrix.png")
    )

    # 7. Save Models
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    head_save_path = models_dir / "cnn_baseline_head.keras"
    head_model.save(head_save_path)
    print(f"[SAVED] Saved baseline head model to: {head_save_path.resolve()}")

    # End-to-end model (takes raw video frames directly)
    e2e_model = build_end_to_end_cnn_baseline(
        cnn_frame_model=extractor.frame_model,
        head_model=head_model,
        sequence_length=seq_len,
        input_shape=(*target_size, 3),
    )
    e2e_save_path = models_dir / "cnn_baseline_end_to_end.keras"
    e2e_model.save(e2e_save_path)
    print(f"[SAVED] Saved end-to-end video baseline model to: {e2e_save_path.resolve()}")

    # 8. Save Metrics
    metrics_summary = {
        "model_name": "CNN_Only_Baseline",
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_samples_total": len(y_val),
        "classes": selected_classes,
        "classification_report": report,
        "epochs_trained": len(history.history["loss"]),
    }
    metrics_path = Path("outputs/logs/cnn_baseline_val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[SAVED] Saved validation metrics to: {metrics_path.resolve()}")

    return metrics_summary


if __name__ == "__main__":
    train_cnn_baseline()
