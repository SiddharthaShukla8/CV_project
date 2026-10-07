"""Evaluate baseline model: macro + per-class metrics, AUC, confusion matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import CLASS_NAMES, make_dataset  # noqa: E402


def _collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset):
    y_true, y_prob = [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    results_dir: Path,
    protocol: Optional[dict] = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = _collect_predictions(model, test_ds)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # One-vs-rest multiclass AUC (macro)
    try:
        auc_macro = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc_macro = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {}
    for name in class_names:
        per_class[name] = {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("M2 Confusion Matrix — InceptionV3 baseline")
    fig.tight_layout()
    cm_path = results_dir / "M2_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paper = (protocol or {}).get("paper_target", {})
    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "auc_macro_ovr": auc_macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
        "class_names": class_names,
        "paper_comparison": {
            "paper_accuracy": paper.get("accuracy", 0.9712),
            "paper_auc": paper.get("auc", 0.9984),
            "paper_recall": paper.get("recall", 0.9659),
            "delta_accuracy": acc - paper.get("accuracy", 0.9712),
            "delta_auc": (None if auc_macro is None else auc_macro - paper.get("auc", 0.9984)),
            "delta_recall": rec_macro - paper.get("recall", 0.9659),
        },
        "limitations_note": (
            "Results are on a public research MRI dataset with the given "
            "Kaggle Training/Testing split (not the paper's 5-fold CV after "
            "augmentation). Metrics are research benchmarks only — no clinical "
            "claims. Always interpret accuracy together with recall/sensitivity "
            "and AUC; per-class scores matter for minority error modes."
        ),
    }
    if protocol:
        metrics["train_protocol"] = {
            k: protocol[k]
            for k in (
                "split_policy",
                "n_train",
                "n_val",
                "n_test",
                "phase1",
                "phase2",
                "head_epochs_ran",
                "finetune_epochs_ran",
                "seed",
            )
            if k in protocol
        }

    out_json = results_dir / "M2_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in (
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr",
        "per_class", "paper_comparison", "limitations_note"
    )}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {cm_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT / "results" / "checkpoints" / "final_inceptionv3.keras"),
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    test_ds, _ = make_dataset(
        args.data_dir,
        split="Testing",
        batch_size=args.batch_size,
        shuffle=False,
        augment_data=False,
    )
    evaluate_model(model, test_ds, list(CLASS_NAMES), Path(args.results_dir))


if __name__ == "__main__":
    main()
