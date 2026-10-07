# M3 — Introduction

## I. Background

Brain tumors are abnormal growths of brain tissue and remain among the most serious neurological conditions. Common primary tumor types studied in public MRI benchmarks include **glioma**, **meningioma**, and **pituitary** tumors; many open datasets also include a **no-tumor** (healthy) class for four-way classification. Magnetic resonance imaging (MRI) is the preferred non-invasive modality for soft-tissue assessment of these lesions. Manual reading of large MRI archives is time-consuming and variable across readers, which has motivated computer-vision research on automated **research** classifiers using convolutional neural networks (CNNs) and transfer learning from ImageNet-pretrained backbones.

This project sits in that research setting. It does **not** make clinical claims: reported numbers are benchmark metrics on a public dataset, and every results write-up must include **recall/sensitivity** and **AUC** alongside accuracy, plus a short limitations note.

## II. Problem statement

**Task:** Multiclass classification of brain MRI slices into four classes — glioma, meningioma, pituitary tumor, and no tumor.

**Data:** The public Kaggle corpus [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 images), combining Figshare, SARTAJ, and Br35H, with the provided `Training/` and `Testing/` folder split.

**Baseline (M2):** Reproduce a transfer-learning pipeline aligned with Gómez-Guzmán et al. (*Electronics*, 2023), whose best reported model was **InceptionV3** (**97.12%** average accuracy, **AUC 0.9984**, **recall 0.9659**).

**Gap this project targets later (M4, not implemented here):** Heavy CNNs can score well while carrying a large parameter budget. Efficiency-aware backbones (EfficientNet/MobileNet family) have reported competitive accuracy at far fewer parameters, but that trade-off was not the selection criterion in the base paper even though EfficientNetB0 was among the models evaluated.

## III. Objectives

1. Survey related CNN / transfer-learning work for brain-tumor MRI classification and fix a reproducible base paper (**M1**).  
2. Reproduce an InceptionV3 baseline on the public 7,023-image four-class set, reporting accuracy **with** macro and per-class precision/recall/F1 and AUC (**M2**).  
3. State a **single-variable**, falsifiable hypothesis *before* changing code: swap InceptionV3 → EfficientNet-B0 under a fixed pipeline (**M3**; implementation deferred to M4).  
4. Keep scope honest: no clinical deployment claims; document dataset split before experiments; note limitations whenever results are reported.

**Out of scope for this milestone packet:** EfficientNet-B0 implementation, ablation tables, deep failure analysis, IEEE camera-ready paper, and viva/defence materials (M4–M5).
