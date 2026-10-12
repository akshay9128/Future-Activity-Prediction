"""
Unified Training Pipeline and Model Benchmarking Orchestrator.

Provides a unified interface to train, evaluate, and benchmark all activity
prediction architectures:
1. CNN-Only Baseline (Temporal Average Pooling)
2. CNN + LSTM (Sequential Recurrent)
3. CNN + LSTM + Attention (Adaptive Temporal Salience)

Generates consolidated comparison metrics, tabular summaries, and multi-metric
visualization charts across all trained models.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_attention import TemporalAttention
from src.models.train_cnn_baseline import train_cnn_baseline
from src.models.train_cnn_lstm import train_cnn_lstm
from src.models.train_cnn_lstm_attention import train_cnn_lstm_attention

MODEL_METRIC_FILES = {
    "CNN_Only_Baseline": "cnn_baseline_val_metrics.json",
    "CNN_LSTM": "cnn_lstm_val_metrics.json",
    "CNN_LSTM_Attention": "cnn_lstm_attention_val_metrics.json",
}

MODEL_ARTIFACT_FILES = {
    "CNN_Only_Baseline": "cnn_baseline_head.keras",
    "CNN_LSTM": "cnn_lstm_head.keras",
    "CNN_LSTM_Attention": "cnn_lstm_attention_head.keras",
}

MODEL_DISPLAY_NAMES = {
    "CNN_Only_Baseline": "CNN Baseline (Avg Pool)",
    "CNN_LSTM": "CNN + LSTM (Recurrent)",
    "CNN_LSTM_Attention": "CNN + LSTM + Attention",
}


def get_model_parameters(model_file: Path) -> int:
    """Load a saved Keras model and return its parameter count."""
    if not model_file.exists():
        return 0
    try:
        model = tf.keras.models.load_model(model_file, compile=False)
        return int(model.count_params())
    except Exception as e:
        print(f"[WARN] Could not inspect parameters from {model_file}: {e}")
        return 0


def generate_comparison_summary(
    logs_dir: Path = Path("outputs/logs"),
    models_dir: Path = Path("outputs/models"),
) -> Dict[str, Any]:
    """
    Aggregate validation metrics and parameter counts across all trained models,
    compile a comparative dictionary, and save outputs/logs/model_comparison_val.json.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_records: List[Dict[str, Any]] = []

    for internal_name, json_filename in MODEL_METRIC_FILES.items():
        json_path = logs_dir / json_filename
        if not json_path.exists():
            print(f"[WARN] Missing metrics file: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        keras_filename = MODEL_ARTIFACT_FILES.get(internal_name, "")
        keras_path = models_dir / keras_filename
        param_count = get_model_parameters(keras_path)

        clf_rep = data.get("classification_report", {})
        macro_avg = clf_rep.get("macro avg", {})
        weighted_avg = clf_rep.get("weighted avg", {})

        record = {
            "model_key": internal_name,
            "display_name": MODEL_DISPLAY_NAMES.get(internal_name, internal_name),
            "val_loss": round(float(data.get("val_loss", 0.0)), 4),
            "val_accuracy": round(float(data.get("val_accuracy", 0.0)), 4),
            "val_accuracy_pct": round(float(data.get("val_accuracy", 0.0)) * 100, 2),
            "macro_precision": round(float(macro_avg.get("precision", 0.0)), 4),
            "macro_recall": round(float(macro_avg.get("recall", 0.0)), 4),
            "macro_f1": round(float(macro_avg.get("f1-score", 0.0)), 4),
            "weighted_f1": round(float(weighted_avg.get("f1-score", 0.0)), 4),
            "trainable_parameters": param_count,
            "epochs_trained": data.get("epochs_trained", 0),
            "val_samples_total": data.get("val_samples_total", 0),
        }
        summary_records.append(record)

    comparison_dict = {
        "timestamp": "2026-10-12",
        "models": summary_records,
    }

    out_file = logs_dir / "model_comparison_val.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_dict, f, indent=2)
    print(f"[SAVED] Saved model comparison summary to: {out_file.resolve()}", flush=True)

    return comparison_dict


def plot_model_comparison(
    comparison_data: Dict[str, Any],
    output_path: Path = Path("outputs/logs/model_comparison_summary.png"),
) -> None:
    """
    Generate a 4-panel multi-metric comparative visualization comparing
    Accuracy, Macro F1, Validation Loss, and Model Parameters across models.
    """
    models = comparison_data.get("models", [])
    if not models:
        print("[WARN] No models available to plot.")
        return

    names = [m["display_name"] for m in models]
    accuracies = [m["val_accuracy_pct"] for m in models]
    f1_scores = [m["macro_f1"] * 100 for m in models]
    losses = [m["val_loss"] for m in models]
    params = [m["trainable_parameters"] for m in models]

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Validation Accuracy
    ax_acc = axes[0, 0]
    bars_acc = ax_acc.bar(names, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax_acc.set_title("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_ylim([90, 102])
    ax_acc.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_acc:
        yval = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Macro F1-Score
    ax_f1 = axes[0, 1]
    bars_f1 = ax_f1.bar(names, f1_scores, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax_f1.set_title("Validation Macro F1-Score (%)", fontsize=12, fontweight="bold")
    ax_f1.set_ylabel("Macro F1 (%)", fontsize=10)
    ax_f1.set_ylim([90, 102])
    ax_f1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_f1:
        yval = bar.get_height()
        ax_f1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Validation Loss
    ax_loss = axes[1, 0]
    bars_loss = ax_loss.bar(names, losses, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax_loss.set_title("Validation Loss (Cross-Entropy)", fontsize=12, fontweight="bold")
    ax_loss.set_ylabel("Validation Loss", fontsize=10)
    ax_loss.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_loss:
        yval = bar.get_height()
        ax_loss.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0005, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Trainable Parameter Count
    ax_params = axes[1, 1]
    bars_params = ax_params.bar(names, [p / 1000.0 for p in params], color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax_params.set_title("Trainable Head Parameters (Thousands)", fontsize=12, fontweight="bold")
    ax_params.set_ylabel("Parameters (k)", fontsize=10)
    ax_params.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, p_count in zip(bars_params, params):
        yval = bar.get_height()
        ax_params.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, f"{p_count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Comparative Model Benchmark: Early Activity Prediction (50% Observed)", fontsize=14, fontweight="bold", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved model comparison chart to: {output_path.resolve()}", flush=True)


def print_comparison_table(comparison_data: Dict[str, Any]) -> None:
    """Print clean ASCII table comparing all models."""
    models = comparison_data.get("models", [])
    if not models:
        return

    header = f"{'Model':<25} | {'Val Acc':<9} | {'Val Loss':<10} | {'Macro F1':<10} | {'Params':<10} | {'Epochs':<6}"
    separator = "-" * len(header)
    print("\n" + separator)
    print("           MODEL BENCHMARK COMPARISON TABLE            ")
    print(separator)
    print(header)
    print(separator)
    for m in models:
        row = (
            f"{m['display_name']:<25} | "
            f"{m['val_accuracy_pct']:>6.2f}%  | "
            f"{m['val_loss']:>8.4f}  | "
            f"{m['macro_f1']*100:>7.2f}%  | "
            f"{m['trainable_parameters']:>9,} | "
            f"{m['epochs_trained']:>6}"
        )
        print(row)
    print(separator + "\n")


def run_training_pipeline(
    models: Optional[List[str]] = None,
    config_path: Path | str = "configs/default_config.yaml",
    skip_training: bool = False,
) -> Dict[str, Any]:
    """
    Execute training pipeline for specified models and generate benchmark reports.

    Args:
        models: List of model identifiers ('all', 'cnn_baseline', 'cnn_lstm', 'cnn_lstm_attention').
        config_path: Path to configuration YAML.
        skip_training: If True, skips training execution and only regenerates summaries.

    Returns:
        Dict[str, Any]: Benchmark summary dictionary.
    """
    valid_models = ["cnn_baseline", "cnn_lstm", "cnn_lstm_attention"]
    if models is None or "all" in models:
        models_to_run = valid_models
    else:
        models_to_run = [m for m in models if m in valid_models]

    if not skip_training:
        print(f"[PIPELINE] Starting training for: {', '.join(models_to_run)}", flush=True)
        if "cnn_baseline" in models_to_run:
            print("\n>>> Running CNN Baseline Training...", flush=True)
            train_cnn_baseline(config_path)

        if "cnn_lstm" in models_to_run:
            print("\n>>> Running CNN + LSTM Training...", flush=True)
            train_cnn_lstm(config_path)

        if "cnn_lstm_attention" in models_to_run:
            print("\n>>> Running CNN + LSTM + Attention Training...", flush=True)
            train_cnn_lstm_attention(config_path)

    # Compile benchmark summary and visualization
    summary_data = generate_comparison_summary()
    plot_model_comparison(summary_data)
    print_comparison_table(summary_data)

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline and Benchmarking for Activity Prediction."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "cnn_baseline", "cnn_lstm", "cnn_lstm_attention"],
        help="Specify which model architecture to train ('all' trains all three).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training and re-aggregate already saved model metrics.",
    )

    args = parser.parse_args()
    run_training_pipeline(
        models=[args.model],
        config_path=args.config,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
