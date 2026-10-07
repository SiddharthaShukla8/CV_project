"""Data loading for brain-tumor MRI baseline (InceptionV3, 299x299)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMG_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE

# Module-level layers so tf.data map graph stays stable
_ROTATE = tf.keras.layers.RandomRotation(0.028, fill_mode="nearest")
_ZOOM = tf.keras.layers.RandomZoom(0.1, fill_mode="nearest")


def _resolve_data_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    if (root / "Training").is_dir() and (root / "Testing").is_dir():
        return root
    # Zenodo / Kaggle sometimes nest one extra folder
    for child in root.iterdir():
        if child.is_dir() and (child / "Training").is_dir():
            return child
    raise FileNotFoundError(
        f"Expected Training/ and Testing/ under {data_dir}"
    )


def inception_preprocess(image: tf.Tensor) -> tf.Tensor:
    """Keras InceptionV3 preprocessing: RGB float in [-1, 1]."""
    return tf.keras.applications.inception_v3.preprocess_input(image)


def augment(image: tf.Tensor) -> tf.Tensor:
    """Light train-only augmentation: flip, small rotation, small zoom."""
    image = tf.image.random_flip_left_right(image)
    image = _ROTATE(image, training=True)
    image = _ZOOM(image, training=True)
    return image


def _decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image


def _paths_and_labels(split_dir: Path) -> Tuple[tf.Tensor, tf.Tensor]:
    paths, labels = [], []
    for idx, name in enumerate(CLASS_NAMES):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in sorted(class_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                paths.append(str(p))
                labels.append(idx)
    return tf.constant(paths), tf.constant(labels, dtype=tf.int32)


def make_dataset(
    data_dir: str | Path,
    split: str = "Training",
    batch_size: int = 16,
    shuffle: bool = True,
    augment_data: bool = False,
    seed: int = 42,
    validation_split: float | None = None,
    subset: str | None = None,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a tf.data pipeline from the Kaggle folder layout.

    If validation_split is set (e.g. 0.2) and split=='Training', uses a
    deterministic 80/20 split of Training/ (subset='training'|'validation').
    Testing/ is always used as the held-out test set (no augmentation).
    """
    root = _resolve_data_root(data_dir)
    split_dir = root / ("Training" if split.lower().startswith("train") else "Testing")
    paths, labels = _paths_and_labels(split_dir)
    n = int(tf.shape(paths)[0])

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if validation_split is not None:
        if subset not in {"training", "validation"}:
            raise ValueError("subset must be 'training' or 'validation'")
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=False)
        val_count = int(n * validation_split)
        train_count = n - val_count
        if subset == "training":
            ds = ds.skip(val_count)
            n = train_count
        else:
            ds = ds.take(val_count)
            n = val_count
            shuffle = False
            augment_data = False

    if shuffle:
        ds = ds.shuffle(min(n, 2000), seed=seed, reshuffle_each_iteration=True)

    def _map(path, label):
        image = _decode_and_resize(path)
        if augment_data:
            image = augment(image)
        image = inception_preprocess(image)
        return image, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, n


def get_train_val_test(
    data_dir: str | Path,
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """Return train / val / test datasets and size metadata."""
    train_ds, n_train = make_dataset(
        data_dir,
        split="Training",
        batch_size=batch_size,
        shuffle=True,
        augment_data=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    val_ds, n_val = make_dataset(
        data_dir,
        split="Training",
        batch_size=batch_size,
        shuffle=False,
        augment_data=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )
    test_ds, n_test = make_dataset(
        data_dir,
        split="Testing",
        batch_size=batch_size,
        shuffle=False,
        augment_data=False,
        seed=seed,
    )
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "class_names": list(CLASS_NAMES),
        "img_size": list(IMG_SIZE),
        "batch_size": batch_size,
        "val_split_of_training": val_split,
        "seed": seed,
        "split_policy": (
            "Official Kaggle Training/Testing folders. "
            f"From Training/, hold out {val_split:.0%} as validation "
            f"(seed={seed}); Testing/ is the final test set."
        ),
    }
    return train_ds, val_ds, test_ds, meta
