# Brain Tumor MRI Classification — Computer Vision Course Project (M1–M3)

[![Milestones](https://img.shields.io/badge/Milestones-M1%20%7C%20M2%20%7C%20M3%20Complete-success)](docs/)
[![Model](https://img.shields.io/badge/Baseline-InceptionV3%20Transfer%20Learning-blue)](results/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.81%25-brightgreen)](results/M2_metrics.json)
[![AUC](https://img.shields.io/badge/Macro%20AUC-0.9835-blueviolet)](results/M2_metrics.json)

---

## 📌 Executive Status & Milestone Tracker

| Milestone | Focus Area | Status | Deliverables |
|---|---|---|---|
| **M1** | Literature Survey & Reproducibility Audit | **Completed** | `docs/M1_literature_survey.md`, `docs/M1_reproducibility_checklist.md` |
| **M2** | InceptionV3 Baseline Reproduction | **Completed** | Model Checkpoint, `results/M2_metrics.json`, `results/M2_confusion_matrix.png`, `results/M2_training_curves.png` |
| **M3** | Problem Formulation & Hypothesis | **Completed** | `docs/M3_introduction.md`, `docs/M3_hypothesis.md` (EfficientNet-B0 **not** implemented) |
| **M4 / M5** | Efficiency-Aware Modeling (EfficientNet-B0) | **Awaiting Approval** | **Intentionally not started — awaiting go-ahead.** |

---

## 📊 M2 Baseline Reproduction Results

### Overall Performance Metrics (1,600 Independent Test Slices)

| Metric | Our Reproduction | Base Paper Target (Gómez-Guzmán et al. 2023) | Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **90.81%** (0.9081) | **97.12%** | **-6.31%** |
| **Macro Recall (Sensitivity)** | **90.81%** (0.9081) | **96.59%** | **-5.78%** |
| **Macro F1-Score** | **90.58%** (0.9058) | — | — |
| **Macro Precision** | **90.97%** (0.9097) | — | — |
| **Macro AUC (One-vs-Rest)** | **0.9835** (0.9835) | **0.9984** | **-0.015** |

### Per-Class Evaluation Breakdown

Evaluated on the balanced test set ($N = 1,600$, 400 slices per class):

| Tumor Class | Precision | Recall (Sensitivity) | F1-Score | Support | Clinical / Diagnostic Observation |
|---|:---:|:---:|:---:|:---:|---|
| **`glioma`** | 93.81% | **75.75%** | 83.82% | 400 | **Weakest class:** Infiltrative, diffuse boundaries lead to confusion with meningioma. |
| **`meningioma`** | 84.73% | **88.75%** | 86.69% | 400 | Extra-axial dural lesions; moderate false positives from axial glioma slices. |
| **`notumor`** | 91.72% | **99.75%** | 95.57% | 400 | **Near-perfect identification:** 399/400 healthy scans correctly classified. |
| **`pituitary`** | 93.62% | **99.00%** | 96.23% | 400 | **Highly distinctive:** 396/400 correct due to prominent sellar anatomical features. |

### Confusion Matrix ($4 \times 4$)

```text
               Predicted:
               glioma   meningioma   notumor   pituitary
True glioma       303           60        30           7
True meningioma    19          355         6          20
True notumor        0            1       399           0
True pituitary      1            3         0         396
```

### 🔍 Honest Gap Analysis & Protocol Differences
- **Cross-Validation vs. Fixed Test Set:** The base paper utilizes a 5-fold cross-validation scheme evaluated on heavily augmented data. Our reproduction enforces an independent, held-out test split (1,600 images) to strictly prevent test-set data leakage.
- **Compute & Epoch Budget:** Trained on CPU for 13 total epochs (8 head + 5 fine-tuning). Despite the limited epoch budget, the baseline achieves **0.9835 AUC**, confirming strong separation capability across tumor subtypes.
- **No Clinical Claims:** This project serves as a computer vision research benchmark. All evaluations report Sensitivity/Recall and AUC alongside Accuracy.

---

## 📈 2-Phase Training Progression

### Model Architecture Parameters
- **Backbone:** ImageNet-pretrained `InceptionV3` (Input: $299 \times 299 \times 3$)
- **Classification Head:** `GlobalAveragePooling2D` $\rightarrow$ `Dense(256, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(4, Softmax)`
- **Total Parameters:** $22,328,356$ ($85.18\text{ MB}$)
- **Phase 1 Trainable Params:** $525,572$ ($2.00\text{ MB}$) | Non-Trainable: $21,802,784$
- **Phase 2 Trainable Params:** $\sim 11.5\text{ MB}$ (Top 50 backbone layers unfrozen)

### Epoch Progression Log

```text
=== Phase 1: Train Classification Head (Backbone Frozen, Adam lr=1e-3) ===
Epoch 1/8 — 1169s — Train Acc: 76.85% | Train Loss: 0.6072 | Val Acc: 88.84% | Val Loss: 0.3345
Epoch 2/8 — 1167s — Train Acc: 85.25% | Train Loss: 0.3967 | Val Acc: 88.30% | Val Loss: 0.3319
Epoch 3/8 — 1049s — Train Acc: 86.29% | Train Loss: 0.3574 | Val Acc: 87.95% | Val Loss: 0.3150
Epoch 4/8 — 1042s — Train Acc: 87.46% | Train Loss: 0.3328 | Val Acc: 89.29% | Val Loss: 0.3264
Epoch 5/8 — 1077s — Train Acc: 88.15% | Train Loss: 0.3064 | Val Acc: 88.48% | Val Loss: 0.3009
Epoch 6/8 — 1044s — Train Acc: 89.04% | Train Loss: 0.2950 | Val Acc: 91.07% | Val Loss: 0.2577
Epoch 7/8 — 1070s — Train Acc: 89.93% | Train Loss: 0.2827 | Val Acc: 90.98% | Val Loss: 0.2471
Epoch 8/8 — 1054s — Train Acc: 90.67% | Train Loss: 0.2493 | Val Acc: 88.84% | Val Loss: 0.2682

=== Phase 2: Fine-Tune Top 50 InceptionV3 Blocks (Adam lr=1e-5) ===
Epoch 1/5 — 1313s — Train Acc: 82.63% | Train Loss: 0.4796 | Val Acc: 89.82% | Val Loss: 0.2423
Epoch 2/5 — 1095s — Train Acc: 90.36% | Train Loss: 0.2590 | Val Acc: 92.41% | Val Loss: 0.1947
Epoch 3/5 —  583s — Train Acc: 91.90% | Train Loss: 0.2125 | Val Acc: 93.39% | Val Loss: 0.1661
Epoch 4/5 —  581s — Train Acc: 93.79% | Train Loss: 0.1815 | Val Acc: 94.38% | Val Loss: 0.1435
Epoch 5/5 —  577s — Train Acc: 94.89% | Train Loss: 0.1414 | Val Acc: 94.91% | Val Loss: 0.1266
```

---

## 🗂️ Dataset Split & Integrity Policy

**Dataset Source:** Kaggle [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (combining Figshare, SARTAJ, and Br35H).  
**Local Mirror:** Zenodo [10.5281/zenodo.12735702](https://doi.org/10.5281/zenodo.12735702).

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total Slices |
|---|:---:|:---:|:---:|:---:|:---:|
| `Training/` (Total) | 1,321 | 1,339 | 1,595 | 1,457 | **5,712** |
| ↳ Train (80% holdout, `seed=42`) | 1,056 | 1,071 | 1,276 | 1,165 | **4,480** |
| ↳ Val (20% holdout, `seed=42`) | 265 | 268 | 319 | 292 | **1,120** |
| `Testing/` (Independent Evaluation) | 300 (400*) | 306 (400*) | 405 (400*) | 300 (400*) | **1,311 (1,600*)** |
| **Total Corpus** | — | — | — | — | **7,023** |

*\*Zenodo evaluation set incorporates 400 slices per class for balanced test distribution.*

---

## 🔬 M3 Pre-Registered Hypothesis

> **Hypothesis:** Swapping only the backbone from **InceptionV3** (~23.85M params) to **EfficientNet-B0** (~5.33M params) while holding all other variables constant (same 2-phase LR schedule, identical head architecture, matching $299 \times 299$ preprocessing, and fixed `seed=42` data split) will maintain Macro F1 and Macro Recall within **$\pm 1\text{–}2\%$** of the M2 baseline while cutting total model parameters by **$\sim 4.5\times$**.

*Status: Prediction documented in `docs/M3_hypothesis.md` prior to any M4 implementation.*

---

## 🚀 Setup & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/SiddharthaShukla8/CV_project.git
cd CV_project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Acquisition
```bash
# Option A: Kaggle API
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip

# Option B: Zenodo Mirror
curl -L -o data/brain-tumor-mri-dataset.zip "https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1"
unzip -q data/brain-tumor-mri-dataset.zip -d data
```

### 3. Baseline Training & Evaluation
```bash
export MPLCONFIGDIR=$PWD/.mplconfig
python -m src.train_baseline \
  --data-dir data \
  --results-dir results \
  --batch-size 16 \
  --head-epochs 8 \
  --finetune-epochs 5 \
  --seed 42
```

---

## 📁 Repository Artifacts & Documentation

```text
CV_project/
├── Brain_Tumor_MRI_Classification_Presentation.pdf  # 6-page landscape executive presentation
├── generate_presentation.py                         # PDF generator script
├── docs/
│   ├── M1_literature_survey.md                      # 11-paper literature synthesis
│   ├── M1_reproducibility_checklist.md              # Gómez-Guzmán audit checklist
│   ├── M3_introduction.md                           # Problem formulation & clinical background
│   └── M3_hypothesis.md                             # Single-variable M4 prediction
├── notebooks/
│   └── M2_baseline_reproduction.ipynb               # 12-cell interactive notebook
├── results/
│   ├── M2_metrics.json                              # Full quantitative evaluation metrics
│   ├── M2_confusion_matrix.png                      # 4x4 Confusion matrix heatmap
│   ├── M2_training_curves.png                       # Phase 1 & 2 loss/accuracy curves
│   └── checkpoints/
│       └── final_inceptionv3.keras                  # Saved trained baseline weights (145 MB)
├── src/
│   ├── data_loader.py                               # Preprocessing, normalization & tf.data pipeline
│   ├── train_baseline.py                            # 2-phase InceptionV3 training loop
│   └── evaluate.py                                  # Metric reporting & visualization generation
├── requirements.txt
└── README.md
```