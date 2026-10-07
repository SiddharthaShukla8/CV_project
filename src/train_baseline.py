"""Train InceptionV3 baseline for 4-class brain-tumor MRI classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import CLASS_NAMES, get_train_val_test  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402


def build_model(num_classes: int = 4, dropout: float = 0.3) -> tf.keras.Model:
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        name="inception_v3",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="inceptionv3_baseline")


def plot_history(history_dicts: list, out_path: Path) -> None:
    keys = ["accuracy", "val_accuracy", "loss", "val_loss"]
    merged = {k: [] for k in keys}
    for h in history_dicts:
        for k in keys:
            merged[k].extend(h.get(k, []))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(merged["accuracy"], label="train")
    axes[0].plot(merged["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(merged["loss"], label="train")
    axes[1].plot(merged["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("M2 Baseline — InceptionV3 (frozen head → fine-tune)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def unfreeze_top_blocks(model: tf.keras.Model, n_unfreeze: int = 50) -> None:
    """Unfreeze the last `n_unfreeze` layers of the InceptionV3 backbone."""
    base = model.get_layer("inception_v3")
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers[-n_unfreeze:]:
        layer.trainable = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, meta = get_train_val_test(
        args.data_dir, batch_size=args.batch_size, val_split=0.2, seed=args.seed
    )
    print(json.dumps(meta, indent=2))

    model = build_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_head = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("\n=== Phase 1: train classification head (backbone frozen) ===")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks_head,
    )

    print("\n=== Phase 2: fine-tune top InceptionV3 blocks (lr=1e-5) ===")
    unfreeze_top_blocks(model, n_unfreeze=50)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_inceptionv3.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=callbacks_ft,
    )

    plot_history([h1.history, h2.history], results_dir / "M2_training_curves.png")
    model.save(ckpt_dir / "final_inceptionv3.keras")

    protocol = {
        **meta,
        "framework": "TensorFlow/Keras",
        "backbone": "InceptionV3 ImageNet pretrained, include_top=False",
        "head": "GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(4, softmax)",
        "phase1": "backbone frozen, Adam 1e-3",
        "phase2": "unfreeze last 50 backbone layers, Adam 1e-5",
        "head_epochs_ran": len(h1.history["loss"]),
        "finetune_epochs_ran": len(h2.history["loss"]),
        "augmentation": "train only: horizontal flip, ~±10° rotation, ±10% zoom",
        "paper_target": {
            "accuracy": 0.9712,
            "auc": 0.9984,
            "recall": 0.9659,
        },
    }
    with open(results_dir / "M2_train_protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)

    if not args.skip_eval:
        evaluate_model(
            model,
            test_ds,
            class_names=list(CLASS_NAMES),
            results_dir=results_dir,
            protocol=protocol,
        )


if __name__ == "__main__":
    main()
