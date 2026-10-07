# M1 — Reproducibility Checklist

**Base paper:** Gómez-Guzmán et al., “Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks,” *Electronics*, 12(4), 955, 2023.  
DOI: [10.3390/electronics12040955](https://doi.org/10.3390/electronics12040955)

| Checklist item | Answer |
|---|---|
| Public code? | **Not found for the official paper.** Searched GitHub, Papers With Code, and the MDPI article page (Oct 2026): no author-linked repository for Gómez-Guzmán et al. (2023). Equivalent public implementations of this **same Kaggle dataset + InceptionV3 / TL pipeline** do exist, e.g. [Laksopan23/Brain-Tumor-Detection-Models](https://github.com/Laksopan23/Brain-Tumor-Detection-Models) (`InceptionV3TL.ipynb`) and [somenath203/Multiclass-Brain-Tumor-Classification-using-TensorFlow](https://github.com/somenath203/Multiclass-Brain-Tumor-Classification-using-TensorFlow). Dataset author’s related ResNet50 notebook: [masoudnick/Brain-Tumor-MRI-Classification](https://github.com/masoudnick/Brain-Tumor-MRI-Classification). |
| Dataset public & size? | **Yes.** Kaggle [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset): **7,023** MRI images (Figshare + SARTAJ + Br35H), 4 classes, pre-split into `Training/` and `Testing/`. **Confirmed locally** via Zenodo mirror [10.5281/zenodo.12735702](https://doi.org/10.5281/zenodo.12735702): zip **149 MB** (listing 155.8 MB); unpacked `Training/`+`Testing/` ≈ **176 MB** images (+ zip ⇒ ~325 MB in `data/`). Counts: glioma 1321/300, meningioma 1339/306, notumor 1595/405, pituitary 1457/300 (train/test). |
| Fits free GPU? | **Yes.** InceptionV3 at **299×299**, batch **16–32**, fits Colab/Kaggle free **T4** (paper itself used Colab with a larger A100). This reproduction also runs on CPU (slower). |
| Last commit / usable? | **N/A for official code** (none found). Community repos above are usable as references (e.g. Laksopan23 last activity ~2024–2025; somenath203 TensorFlow multiclass repo). **This project’s own M2 code** is the reproduction vehicle. |
| Metric defined? | **Yes.** Paper Table 5 (InceptionV3 average over 5-fold): **Accuracy 97.12% (0.9712)**, **Precision 0.9797**, **Recall 0.9659**, **Specificity 0.9998**, **AUC 0.9984**, Loss 0.0796. Accuracy is primary in the abstract; AUC and recall are reported — we require the same suite (plus macro/per-class F1) for M2. |

## Honest notes

- The paper uses **k-fold CV** after augmentation and reports averaged metrics; our M2 uses the **given Kaggle Training/Testing split** (documented in `README.md`) for a single, re-runnable baseline — expect a small gap vs 97.12% / 0.9984 / 0.9659.
- Parameter count cited in paper Table 7 for InceptionV3: **~23.9M**. We verify counts in M3 against Keras applications.
