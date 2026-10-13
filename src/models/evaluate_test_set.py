"""
Held-Out Test Set Evaluation and Generalization Analysis Module.

Evaluates all trained activity prediction architectures on the unseen, held-out
test split (99 UCF-101 videos):
1. CNN-Only Baseline (Temporal Average Pooling)
2. CNN + LSTM (Sequential Recurrent)
3. CNN + LSTM + Attention (Adaptive Temporal Salience)

Extracts test features, computes classification metrics (Accuracy, Macro F1, Loss,
Confusion Matrices, Per-Class Breakdown), assesses the generalization gap against
validation performance, and produces publication-quality comparative visualizations.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
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
from src.models.temporal_attention import TemporalAttention
from src.utils.config import load_config

MODEL_CONFIGS = [
    {
        "key": "CNN_Only_Baseline",
        "display_name": "CNN Baseline (Avg Pool)",
        "model_file": "cnn_baseline_head.keras",
        "val_metrics_file": "cnn_baseline_val_metrics.json",
        "color": "#2b5c8f",
    },
    {
        "key": "CNN_LSTM",
        "display_name": "CNN + LSTM (Recurrent)",
        "model_file": "cnn_lstm_head.keras",
        "val_metrics_file": "cnn_lstm_val_metrics.json",
        "color": "#d95f02",
    },
    {
        "key": "CNN_LSTM_Attention",
        "display_name": "CNN + LSTM + Attention",
        "model_file": "cnn_lstm_attention_head.keras",
        "val_metrics_file": "cnn_lstm_attention_val_metrics.json",
        "color": "#7570b3",
    },
]


def load_test_features(
    config: Dict[str, Any],
    cache_dir: Path = Path("outputs/cache"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract or load cached features for the held-out test split.

    Returns:
        Tuple: (x_test, y_test, selected_classes)
    """
    manifest_path = Path(config["paths"]["manifest_csv"])
    manifest_df = load_manifest(manifest_path)
    cfg_classes = config["dataset"].get("selected_classes")
    if isinstance(cfg_classes, list):
        selected_classes = sorted(cfg_classes)
    else:
        selected_classes = sorted(manifest_df["class_name"].unique())

    seq_len = config["preprocessing"]["sequence_length"]
    target_size = tuple(config["preprocessing"]["target_size"])
    obs_ratio = config["preprocessing"]["observation_ratio"]

    extractor = CNNFeatureExtractor(
        backbone_name=config["model"]["backbone"],
        input_shape=(*target_size, 3),
        trainable=False,
        sequence_length=seq_len,
    )

    max_test = config["dataset"].get("max_test_per_class", None)
    x_test, y_test = get_or_extract_split_features(
        manifest_df=manifest_df,
        split_name="test",
        feature_extractor=extractor,
        cache_dir=cache_dir,
        sequence_length=seq_len,
        target_size=target_size,
        observation_ratio=obs_ratio,
        max_samples_per_class=max_test,
    )

    if len(np.unique(y_test)) == 5 and len(selected_classes) != 5:
        selected_classes = ["ApplyEyeMakeup", "Basketball", "Biking", "JumpingJack", "WalkingWithDog"]

    return x_test, y_test, selected_classes


def evaluate_single_model(
    model_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_classes: List[str],
) -> Dict[str, Any]:
    """
    Evaluate a saved model head on the test set and compute metrics.
    """
    model = tf.keras.models.load_model(model_path, compile=False)

    # Measure inference timing
    start_time = time.perf_counter()
    probs = model.predict(x_test, verbose=0)
    infer_duration = time.perf_counter() - start_time
    avg_infer_ms = round((infer_duration / max(len(x_test), 1)) * 1000, 2)

    preds = np.argmax(probs, axis=1)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(loss_fn(y_test, probs).numpy())
    test_acc = float(np.mean(preds == y_test))

    # Top-5 Accuracy
    if probs.shape[1] >= 5:
        top5_preds = np.argsort(probs, axis=1)[:, -5:]
        top5_correct = np.array([y in top5 for y, top5 in zip(y_test, top5_preds)])
        top5_acc = float(np.mean(top5_correct))
    else:
        top5_acc = test_acc

    labels = list(range(len(selected_classes)))
    cm = confusion_matrix(y_test, preds, labels=labels)
    clf_report = classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=selected_classes,
        output_dict=True,
        zero_division=0,
    )

    model_size_mb = round(Path(model_path).stat().st_size / (1024 * 1024), 2)

    return {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_accuracy_pct": round(test_acc * 100, 2),
        "top5_accuracy": round(top5_acc, 4),
        "top5_accuracy_pct": round(top5_acc * 100, 2),
        "inference_ms_per_video": avg_infer_ms,
        "model_size_mb": model_size_mb,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "predictions": preds.tolist(),
        "parameter_count": int(model.count_params()),
    }


def plot_side_by_side_confusion_matrices(
    evaluation_records: List[Dict[str, Any]],
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Plot a 3-panel side-by-side figure showing test confusion matrices for all models.
    """
    num_models = len(evaluation_records)
    fig, axes = plt.subplots(1, num_models, figsize=(6.0 * num_models, 5.2), sharey=True)
    if num_models == 1:
        axes = [axes]

    cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Purples]
    n_classes = len(class_names)
    total_samples = int(np.sum(evaluation_records[0]["confusion_matrix"])) if evaluation_records else 0

    for idx, (rec, ax) in enumerate(zip(evaluation_records, axes)):
        cm = np.array(rec["confusion_matrix"])
        cmap = cmaps[idx % len(cmaps)]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(
            f"{rec['display_name']}\nTest Acc: {rec['test_accuracy_pct']:.2f}% | Top-5: {rec.get('top5_accuracy_pct', 0.0):.1f}%",
            fontsize=10,
            fontweight="bold",
        )

        if n_classes <= 15:
            ax.set_xticks(np.arange(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            if idx == 0:
                ax.set_yticks(np.arange(n_classes))
                ax.set_yticklabels(class_names, fontsize=8)
                ax.set_ylabel("Ground Truth Label", fontsize=10)
            else:
                ax.set_yticks(np.arange(n_classes))

            ax.set_xlabel("Predicted Label", fontsize=10)

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
                        fontsize=7,
                    )
        else:
            step = max(1, n_classes // 10)
            ticks = np.arange(0, n_classes, step)
            ax.set_xticks(ticks)
            ax.set_xlabel("Predicted Class Index", fontsize=10)
            if idx == 0:
                ax.set_yticks(ticks)
                ax.set_ylabel("Ground Truth Class Index", fontsize=10)
            else:
                ax.set_yticks(ticks)

    plt.suptitle(
        f"Held-Out Test Set Confusion Matrices ({total_samples} Videos across {n_classes} Classes)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved side-by-side confusion matrices to: {output_path.resolve()}", flush=True)


def plot_test_comparative_metrics(
    evaluation_records: List[Dict[str, Any]],
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Plot comparative metrics comparing Validation vs Test Accuracy, Macro F1,
    Test Loss, and Per-Class F1-scores across models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r["display_name"] for r in evaluation_records]
    val_accs = [r.get("val_accuracy_pct", 0.0) for r in evaluation_records]
    test_accs = [r["test_accuracy_pct"] for r in evaluation_records]
    test_losses = [r["test_loss"] for r in evaluation_records]
    macro_f1s = [r["classification_report"]["macro avg"]["f1-score"] * 100 for r in evaluation_records]
    colors = [r.get("color", "#2171b5") for r in evaluation_records]

    # Panel 1: Validation vs Test Accuracy (Generalization Gap)
    ax1 = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    b1 = ax1.bar(x - width / 2, val_accs, width, label="Validation Accuracy", color="#6baed6", edgecolor="black")
    b2 = ax1.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="#2171b5", edgecolor="black")
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_title("Generalization Gap: Validation vs. Test Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    max_acc = max(val_accs + test_accs) if (val_accs + test_accs) else 100
    ax1.set_ylim([0, min(105, max_acc + 15)])
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=8)
    for b in b2:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel 2: Test Macro F1-Score & Top-5 Accuracy
    ax2 = axes[0, 1]
    top5_accs = [r.get("top5_accuracy_pct", 0.0) for r in evaluation_records]
    b_top1 = ax2.bar(x - width / 2, test_accs, width, label="Top-1 Accuracy", color="#2171b5", edgecolor="black")
    b_top5 = ax2.bar(x + width / 2, top5_accs, width, label="Top-5 Accuracy", color="#2ca02c", edgecolor="black")
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Held-Out Test Top-1 vs. Top-5 Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in b_top1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=8)
    for b in b_top5:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel 3: Test Loss
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, test_losses, color=colors, width=0.5, edgecolor="black")
    ax3.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax3.set_title("Held-Out Test Loss", fontsize=11, fontweight="bold")
    max_loss = max(test_losses) if test_losses else 1.0
    ax3.set_ylim([0, max_loss * 1.25])
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 4: Per-Class F1-Scores / Distribution
    ax4 = axes[1, 1]
    num_classes = len(class_names)
    if num_classes <= 10:
        x_c = np.arange(num_classes)
        w_c = 0.25
        offsets = [-w_c, 0, w_c]
        for idx, r in enumerate(evaluation_records):
            class_f1s = [r["classification_report"][c]["f1-score"] * 100 for c in class_names]
            ax4.bar(x_c + offsets[idx], class_f1s, w_c, label=r["display_name"], color=r["color"], edgecolor="black")
        ax4.set_ylabel("F1-Score (%)", fontsize=10)
        ax4.set_title("Per-Class Test F1-Score Breakdown", fontsize=11, fontweight="bold")
        ax4.set_xticks(x_c)
        ax4.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
        ax4.set_ylim([0, 105])
        ax4.legend(loc="lower right", fontsize=8)
        ax4.grid(axis="y", linestyle=":", alpha=0.6)
    else:
        f1_data = []
        for r in evaluation_records:
            f1s = [
                r["classification_report"][c]["f1-score"] * 100
                for c in class_names
                if c in r["classification_report"]
            ]
            f1_data.append(f1s)
        bp = ax4.boxplot(
            f1_data,
            tick_labels=names,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax4.set_ylabel("F1-Score (%)", fontsize=10)
        ax4.set_title(f"Per-Class F1 Distribution ({num_classes} Classes)", fontsize=11, fontweight="bold")
        ax4.set_ylim([0, 105])
        ax4.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Held-Out Test Set Performance & Generalization Analysis", fontsize=14, fontweight="bold", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved comparative test metrics chart to: {output_path.resolve()}", flush=True)


def print_test_benchmark_table(evaluation_records: List[Dict[str, Any]]) -> None:
    """Print clean ASCII table comparing Test set performance and generalization gaps."""
    header = (
        f"{'Model':<25} | {'Val Acc':<8} | {'Test Acc':<9} | {'Top-5 Acc':<9} | "
        f"{'Gen Gap':<8} | {'Test Loss':<10} | {'Macro F1':<9} | {'Latency':<9} | {'Size':<8}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print("          HELD-OUT TEST SET EVALUATION BENCHMARK          ")
    print(separator)
    print(header)
    print(separator)

    for r in evaluation_records:
        val_acc = r.get("val_accuracy_pct", 0.0)
        test_acc = r["test_accuracy_pct"]
        top5_acc = r.get("top5_accuracy_pct", 0.0)
        gen_gap = val_acc - test_acc
        macro_f1 = r["classification_report"]["macro avg"]["f1-score"] * 100
        latency_str = f"{r.get('inference_ms_per_video', 0.0):.1f} ms"
        size_str = f"{r.get('model_size_mb', 0.0):.1f} MB"

        row = (
            f"{r['display_name']:<25} | "
            f"{val_acc:>6.2f}% | "
            f"{test_acc:>7.2f}% | "
            f"{top5_acc:>7.2f}% | "
            f"{gen_gap:>+6.2f}% | "
            f"{r['test_loss']:>9.4f} | "
            f"{macro_f1:>7.2f}% | "
            f"{latency_str:>9} | "
            f"{size_str:>8}"
        )
        print(row)
    print(separator + "\n")


def run_test_evaluation(
    config_path: Path | str = "configs/default_config.yaml",
    models_dir: Path = Path("outputs/models"),
    logs_dir: Path = Path("outputs/logs"),
) -> Dict[str, Any]:
    """
    Main execution routine for held-out test set evaluation.

    Returns:
        Dict: Full test evaluation results dictionary.
    """
    config = load_config(config_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================", flush=True)
    print("      Held-Out Test Set Evaluation Pipeline       ", flush=True)
    print("==================================================", flush=True)

    # 1. Load or extract test features
    x_test, y_test, selected_classes = load_test_features(config)
    print(f"Test Set Samples: {len(y_test)} videos (16 frames each)", flush=True)
    print(f"Feature Shape   : {x_test.shape}", flush=True)
    class_str = ", ".join(selected_classes) if len(selected_classes) <= 5 else f"{len(selected_classes)} classes ({selected_classes[0]} ... {selected_classes[-1]})"
    print(f"Classes ({len(selected_classes)})   : {class_str}", flush=True)
    print("--------------------------------------------------", flush=True)

    # 2. Evaluate each model
    evaluation_records: List[Dict[str, Any]] = []

    for cfg in MODEL_CONFIGS:
        model_file = models_dir / cfg["model_file"]
        if not model_file.exists():
            print(f"[WARN] Model file missing: {model_file}")
            continue

        print(f"\nEvaluating: {cfg['display_name']}...", flush=True)
        res = evaluate_single_model(model_file, x_test, y_test, selected_classes)
        res["model_key"] = cfg["key"]
        res["display_name"] = cfg["display_name"]
        res["color"] = cfg["color"]

        # Read validation metrics for generalization gap computation
        val_file = logs_dir / cfg["val_metrics_file"]
        if val_file.exists():
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            res["val_accuracy"] = round(float(val_data.get("val_accuracy", 0.0)), 4)
            res["val_accuracy_pct"] = round(float(val_data.get("val_accuracy", 0.0)) * 100, 2)
            res["val_loss"] = round(float(val_data.get("val_loss", 0.0)), 4)
            res["generalization_gap_accuracy"] = round(res["val_accuracy_pct"] - res["test_accuracy_pct"], 2)

        evaluation_records.append(res)
        print(f"  -> Test Accuracy: {res['test_accuracy_pct']:.2f}% | Test Loss: {res['test_loss']:.4f}")

    # 3. Save side-by-side confusion matrices
    plot_side_by_side_confusion_matrices(
        evaluation_records,
        selected_classes,
        logs_dir / "test_confusion_matrices.png",
    )

    # 4. Save comparative metrics visualization
    plot_test_comparative_metrics(
        evaluation_records,
        selected_classes,
        logs_dir / "test_metrics_comparison.png",
    )

    # 5. Save structured JSON evaluation summary
    output_summary = {
        "timestamp": "2026-10-12",
        "test_samples_total": len(y_test),
        "classes": selected_classes,
        "models": evaluation_records,
    }
    json_out_path = logs_dir / "test_set_evaluation.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(f"\n[SAVED] Saved test evaluation summary to: {json_out_path.resolve()}", flush=True)

    # 6. Print console benchmark table
    print_test_benchmark_table(evaluation_records)

    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained activity prediction models on held-out test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    run_test_evaluation(config_path=args.config)


if __name__ == "__main__":
    main()
