# M1 — Literature Survey: Brain Tumor MRI Classification

**Scope:** Four-class MRI classification (glioma, meningioma, pituitary, no tumor).  
**Selected base paper:** [5] Gómez-Guzmán et al., *Electronics* 2023.  
**Note:** Original literature notes were not attached to this session; this survey is reconstructed from the cited base paper, the planned M4 references (Reyes & Sánchez; Priyadarshini et al.), and standard works in the same line. Swap in your exact [1]–[11] wording if it differs.

---

## Table I — Surveyed works

| # | Paper | Backbone / method | Dataset | Key reported metric(s) | One-line limitation |
|---|---|---|---|---|---|
| [1] | Cheng et al., *PLoS ONE*, 2015 | Handcrafted features (intensity / GLCM / BoW) + SVM; tumor-region augmentation & ring partition | Figshare CE-MRI, 3,064 slices, 3 classes | Up to **91.28%** accuracy (BoW + ring partition) | Pre-deep-learning pipeline; no end-to-end CNN; 3 classes only (no healthy class) |
| [2] | Abiwinanda et al., IUPESM, 2018 | Shallow custom CNN (1 conv + pool + FC) | Figshare, 3,064 images, 3 classes | Train **98.51%**, val **84.19%** | Large train–val gap; no AUC/recall emphasis; 3-class only |
| [3] | Sultan et al., *IEEE Access*, 2019 | Custom multi-class CNN | Two public MRI sets (tumor-type / glioma-grade) | **96.13%** / **98.7%** accuracy on the two sets | Accuracy-centric reporting; limited efficiency / explainability analysis |
| [4] | Badža & Barjaktarović, *Applied Sciences*, 2020 | Custom CNN | Figshare CE-MRI, 3 classes | High accuracy (~**96–97%** range in reported CV) | Single-dataset dependence; no combined Figshare+SARTAJ+Br35H mix |
| **[5]** | **Gómez-Guzmán et al., *Electronics*, 2023 (BASE)** | Generic CNN + TL: ResNet50, **InceptionV3**, InceptionResNetV2, Xception, MobileNetV2, EfficientNetB0 | Msoud / Kaggle combined set, **7,023** images, **4 classes** | **InceptionV3: Acc 97.12%, AUC 0.9984, Recall 0.9659** | No public official code; heavy TL models; limited efficiency-focused selection despite including MobileNet/EfficientNet |
| [6] | Nayak et al., 2022 | EfficientNet variant + min–max normalization | ~3,260 MRI, 4 classes | Test accuracy **~98.78%** | Smaller single corpus than the 7k combined set; limited cross-dataset validation |
| [7] | Raza et al., *Computers*, 2022 (DeepTumorNet) | Hybrid GoogLeNet-based deep model | Public 3-class MRI | Acc **99.6%**, Recall **100%**, F1 **99.66%** | Near-ceiling metrics on smaller/easier setups; limited efficiency analysis |
| [8] | Shah et al., *IEEE Access*, 2022 | Fine-tuned EfficientNet | Public MRI tumor set | Strong accuracy with EfficientNet fine-tuning | Dataset / split differences hinder direct comparison to [5] |
| [9] | Reyes & Sánchez, *Heliyon*, 2024 | VGG, ResNet, MobileNet, EfficientNet, ConvNeXt; scratch / aug / TL / fine-tune | Two MRI sets (>3k images), 4 classes | Best ~**98.7%** acc; MobileNet ~3.2M / EfficientNetB0 ~5.9M params vs VGG **171M+** | Different corpus than Msoud 7,023; still limited clinical external validation |
| [10] | Priyadarshini et al., *e-Prime*, 2024 | Fine-tuned **EfficientNetV2S** (ECNN) | Three datasets / multigrade settings | Avg test Acc **98.48%**, Recall **98%**, Precision **98.5%**, Sensitivity **98.71%** | Multi-dataset claim but not the exact Msoud pipeline of [5]; Grad-CAM / XAI not the paper’s primary focus in all reports |
| [11] | Hybrid / XAI EfficientNet lines (e.g. Sci. Reports EfficientNet+Grad-CAM; related EfficientNetV2S+Grad-CAM studies, 2023–2025) | EfficientNet / EfficientNetV2 + Grad-CAM | Figshare or 4-class Kaggle MRI | Often **≥99%** accuracy with saliency maps | High scores on familiar public sets; weak multi-site generalization & inconsistent metric suites |

---

## Research gaps (recurring)

Across these works, four gaps keep showing up. **(1) Small or imbalanced datasets** remain common—Figshare’s ~3k three-class set and uneven class counts still dominate many high scores. **(2) Single-dataset dependence** makes it hard to know whether a method transfers beyond one Kaggle/Figshare split. **(3) Inconsistent metrics**—many papers lead with accuracy alone and omit recall/sensitivity, AUC, or per-class breakdowns that matter for minority tumor types. **(4) Limited efficiency and explainability analysis**—heavier CNNs are often preferred without a clear accuracy-per-parameter trade-off, and Grad-CAM / failure analysis is still secondary. These gaps motivate keeping Gómez-Guzmán et al. [5] as the reproducible baseline (InceptionV3 on the public 7,023-image four-class set) and later testing a **backbone-only** swap to EfficientNet-B0 under a fixed pipeline (M4).

**Disclaimer:** Results above are research benchmarks on public MRI datasets. This project makes **no clinical claims** and always reports recall/sensitivity and AUC alongside accuracy.
