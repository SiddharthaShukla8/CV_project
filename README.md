# Brain Tumor MRI Classification — CV Course Project (M1–M3)

## Status Report (M1–M3)

| Milestone | Status |
|---|---|
| **M1** | Literature survey + reproducibility checklist — **Done** |
| **M2** | InceptionV3 baseline reproduction — **Training/evaluation complete or in progress** |
| **M3** | Hypothesis + Introduction — **Done** |
| **M4 / M5** | **Intentionally not started — awaiting go-ahead** |

### Current Progress

- **M1 — Literature Survey:** Completed. See `docs/M1_literature_survey.md` and `docs/M1_reproducibility_checklist.md`.
- **M2 — Baseline Reproduction:** ImageNet-pretrained **InceptionV3** using TensorFlow/Keras on the public 7,023-image, four-class MRI dataset.
- Evaluation includes:
  - Accuracy
  - Macro and per-class Precision
  - Macro and per-class Recall
  - Macro and per-class F1
  - AUC
  - Confusion Matrix
- **M3 — Hypothesis + Introduction:** Completed. EfficientNet-B0 has **not** been implemented yet.
- **M4 / M5:** Intentionally not started and awaiting approval.

### Baseline vs. Original Paper

The original paper by Gómez-Guzmán et al. (2023) reports the following target results:

- Accuracy: **97.12%**
- AUC: **0.9984**
- Recall: **0.9659**

An exact match is not required. Any difference is reported honestly in `results/M2_metrics.json`.

The experimental protocol differs from the paper because this project uses the official Kaggle train/test split, a fixed validation split, CPU training, and a limited epoch budget, while the paper uses 5-fold cross-validation with heavier augmentation.

### M4 Hypothesis

> Swapping only the backbone to **EfficientNet-B0** (~5.33M parameters vs. ~23.85M for InceptionV3) should keep macro F1/recall within approximately ±1–2% of the M2 baseline while reducing the number of parameters by roughly 4.5×.

**M4 and M5 have intentionally not been started.**

---

## Dataset Split

**Source:** Kaggle `masoudnickparvar/brain-tumor-mri-dataset`

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset combines images from Figshare, SARTAJ, and Br35H.

### Local Mirror

When Kaggle API credentials were unavailable, the dataset was obtained from the following Zenodo mirror:

https://doi.org/10.5281/zenodo.12735702

The downloaded archive was approximately **149 MB**, with the extracted dataset occupying approximately **325 MB**.

### Dataset Distribution

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| `Training/` | 1321 | 1339 | 1595 | 1457 | **5712** |
| `Testing/` | 300 | 306 | 405 | 300 | **1311** |
| **All** | — | — | — | — | **7023** |

### M2 Data Policy

The following split policy was fixed before training:

1. Use the official Kaggle `Training/` and `Testing/` folders.
2. Do **not** reshuffle or modify the official test set.
3. From `Training/`, hold out **20%** for validation using `seed=42`.
4. Use the remaining **80%** for training.
5. Apply light augmentation to the training data only.
6. Report final metrics exclusively on the official `Testing/` set.

This differs from the paper's **5-fold cross-validation** on an augmented pool, so a small difference in metrics is expected.

---

## Why TensorFlow/Keras?

`tf.keras.applications.InceptionV3` provides ImageNet-pretrained weights and native **299×299** input preprocessing, matching the architecture used in the baseline reproduction.

PyTorch could also be used, but TensorFlow/Keras provides a shorter path to reproducing the transfer-learning setup.

---

## Setup

```bash
cd /path/to/CV-Project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Download Dataset

#### Option A — Kaggle API

Requires `~/.kaggle/kaggle.json`.

```bash
kaggle datasets download \
  -d masoudnickparvar/brain-tumor-mri-dataset \
  -p data \
  --unzip
```

#### Option B — Zenodo Mirror

```bash
curl -L -o data/brain-tumor-mri-dataset.zip \
  "https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1"

unzip -q data/brain-tumor-mri-dataset.zip -d data
```

---

## Train & Evaluate Baseline (M2)

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Set the Matplotlib configuration directory:

```bash
export MPLCONFIGDIR=$PWD/.mplconfig
```

Run the baseline:

```bash
python -m src.train_baseline \
  --data-dir data \
  --results-dir results \
  --batch-size 16 \
  --head-epochs 8 \
  --finetune-epochs 5 \
  --seed 42
```

Alternatively, run:

```text
notebooks/M2_baseline_reproduction.ipynb
```

from top to bottom using Colab, Kaggle, or a local environment.

### Outputs

The training pipeline generates:

```text
results/M2_metrics.json
results/M2_confusion_matrix.png
results/M2_training_curves.png
results/checkpoints/final_inceptionv3.keras
```

`M2_metrics.json` contains:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- AUC
- Experimental limitations

---

## Documentation

| File | Milestone | Description |
|---|---|---|
| `docs/M1_literature_survey.md` | M1 | Literature survey covering 11 papers |
| `docs/M1_reproducibility_checklist.md` | M1 | Reproducibility checklist for the base paper |
| `docs/M3_hypothesis.md` | M3 | Hypothesis defined before EfficientNet implementation |
| `docs/M3_introduction.md` | M3 | Project introduction |

---

## Course Rules Honored

- **Predict before modifying:** M3 hypothesis was documented before any EfficientNet implementation.
- **No clinical claims:** The project is treated as a computer vision classification experiment and does not make clinical claims.
- **Evaluation:** Recall and AUC are reported alongside accuracy and other classification metrics.
- **Reproducibility:** Dataset split, random seed, training configuration, and evaluation protocol are documented.
- **Scope control:** Only **M1–M3** have been completed. M4/M5 remain intentionally unstarted pending approval.