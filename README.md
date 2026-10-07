# Brain Tumor MRI Classification — CV Course Project (M1–M3)

## Status report (M1–M3)

| Milestone | Status |
|---|---|
| **M1** Literature survey + reproducibility checklist | Done — see `docs/M1_*.md` |
| **M2** InceptionV3 baseline reproduction | Training/eval in progress or complete — see `results/M2_metrics.json` |
| **M3** Hypothesis + Introduction (docs only) | Done — EfficientNet-B0 **not** implemented |
| **M4 / M5** | **Intentionally not started — awaiting go-ahead.** |

- **Reproduced:** ImageNet-pretrained **InceptionV3** (TensorFlow/Keras) on the public 7,023-image four-class MRI set, with accuracy + macro/per-class precision/recall/F1 + AUC + confusion matrix.
- **Vs paper (Gómez-Guzmán et al. 2023):** target **97.12% / AUC 0.9984 / recall 0.9659**. Exact match not required; any gap is reported honestly in `results/M2_metrics.json` (different protocol: official Kaggle test split vs paper’s 5-fold CV after heavier augmentation; CPU training / epoch budget).
- **M4 hypothesis (one sentence):** Swapping only the backbone to **EfficientNet-B0** (~5.33M params vs InceptionV3 ~23.85M) should keep macro F1/recall within ±1–2% of the M2 baseline while cutting parameters ~4.5×.
- **M4 and M5 intentionally not started — awaiting go-ahead.**

---

## Dataset split (documented *before* training)

**Source:** Kaggle [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Figshare + SARTAJ + Br35H).  
**Local mirror used when Kaggle API credentials were unavailable:** Zenodo [10.5281/zenodo.12735702](https://doi.org/10.5281/zenodo.12735702) (`brain-tumor-mri-dataset.zip`, **149 MB** download / **~155.8 MB** listed; unpacked under `data/` ≈ **325 MB** with zip retained).

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| `Training/` | 1321 | 1339 | 1595 | 1457 | **5712** |
| `Testing/` | 300 | 306 | 405 | 300 | **1311** |
| **All** | | | | | **7023** |

**Policy used for M2 (fixed):**

1. Use the **official Kaggle `Training/` / `Testing/` folders** (no reshuffle of the test set).
2. From `Training/`, hold out **20%** as validation with `seed=42` (deterministic shuffle then take/skip).
3. Remaining **80%** of `Training/` is used for training with light augmentation only.
4. Final metrics are reported on **`Testing/`** only.

This differs from the paper’s **5-fold CV** on an augmented pool — expect a small metric gap vs Table 5 averages.

---

## Why TensorFlow/Keras

`tf.keras.applications.InceptionV3` provides ImageNet weights and native **299×299** preprocessing matching the architecture the base paper highlights. PyTorch would also work; we stay on Keras for a short path to the reported TL setup.

---

## Setup

```bash
cd /path/to/CV-Project   # or brain-tumor-cv-project/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data (pick one)
# A) Kaggle API (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip

# B) Zenodo mirror (used in this session)
curl -L -o data/brain-tumor-mri-dataset.zip \
  "https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1"
unzip -q data/brain-tumor-mri-dataset.zip -d data
```

## Train & evaluate baseline (M2)

```bash
source .venv/bin/activate
export MPLCONFIGDIR=$PWD/.mplconfig
python -m src.train_baseline --data-dir data --results-dir results \
  --batch-size 16 --head-epochs 8 --finetune-epochs 5 --seed 42
```

Or run `notebooks/M2_baseline_reproduction.ipynb` top-to-bottom (Colab/Kaggle/local).

**Outputs:**

- `results/M2_metrics.json` — accuracy, precision, recall, F1 (macro + per-class), AUC  
- `results/M2_confusion_matrix.png`  
- `results/M2_training_curves.png`  
- `results/checkpoints/final_inceptionv3.keras`

## Docs

| File | Milestone |
|---|---|
| `docs/M1_literature_survey.md` | M1 survey (11 papers) |
| `docs/M1_reproducibility_checklist.md` | M1 checklist for base paper |
| `docs/M3_hypothesis.md` | M3 prediction (before EfficientNet code) |
| `docs/M3_introduction.md` | M3 Introduction |

## Course rules honored

- Predict before modifying (M3 before any EfficientNet code).  
- No clinical claims; recall + AUC with accuracy; limitations note in metrics JSON.  
- Scope stop: **M1–M3 only.**
