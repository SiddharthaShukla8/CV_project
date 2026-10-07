# M3 — Hypothesis & Prediction (written before any EfficientNet code)

**Status:** Prediction locked **before** M4 implementation. This repository currently contains **no EfficientNet-B0 training/evaluation code**.

## One variable

**Backbone only:** replace ImageNet-pretrained **InceptionV3** with ImageNet-pretrained **EfficientNet-B0**, keeping fixed:

- dataset & official Kaggle `Training/` / `Testing/` split policy used in M2  
- train-only augmentation (horizontal flip, small rotation, small zoom)  
- head recipe (GAP → dense → dropout → 4-way softmax)  
- two-phase schedule (freeze backbone → unfreeze top blocks at lower LR)  
- metrics suite (accuracy, macro precision/recall/F1, AUC, per-class breakdown)

## Hypothesis

The M2 baseline uses InceptionV3, a relatively heavy CNN (~**23.85M** parameters with the standard Keras ImageNet classifier head; paper Table 7 lists **~23.9M**). Replacing that backbone with EfficientNet-B0 (~**5.33M** parameters with the Keras ImageNet head; backbone-only ~**4.05M**) while holding the rest of the pipeline fixed is predicted to **maintain or slightly improve** macro recall and macro F1 (within about **±1–2 percentage points** of the M2 baseline), while **substantially cutting** parameter count and expected inference cost.

## Predicted magnitude (falsifiable)

| Quantity | Prediction |
|---|---|
| Macro recall / macro F1 | Within **±1–2%** of the M2 InceptionV3 baseline on the same test split |
| Accuracy / AUC | Same ballpark as baseline; no large collapse expected if fine-tuning matches M2 |
| Parameter count | Drop from **~23.9M → ~5.3M** (~**4.5×** fewer parameters; paper’s 23.9M vs Keras EfficientNetB0 ~5.33M) |
| Inference / train step time | Noticeably lower than InceptionV3 at matched batch size (direction: faster; exact wall-clock depends on hardware) |

**Falsification:** If EfficientNet-B0 under the *same* pipeline drops macro recall or macro F1 by **>2 points** vs M2, the “better accuracy-per-parameter on this exact setup” claim is rejected for this project (even if it still wins on size).

## Mechanism

EfficientNet’s **compound scaling** jointly scales depth, width, and resolution, which Reyes & Sánchez (*Heliyon*, 2024) showed yields a much better accuracy-to-parameter trade-off than older heavy CNNs (e.g. VGG-class models with **171M+** parameters vs MobileNet ~3.2M / EfficientNetB0 ~5.9M in their study). Priyadarshini et al. (*e-Prime*, 2024) further support that fine-tuned EfficientNet-family models (EfficientNetV2S in their work) remain competitive (~98.5% accuracy/precision/recall on their datasets) — used here only as secondary motivation for the fine-tuning recipe, **not** as M4 code.

## Verified parameter counts (Keras / TensorFlow, this environment)

| Model | `include_top=True` (ImageNet head) | `include_top=False` backbone |
|---|---|---|
| InceptionV3 | **23,851,784** (~23.85M) | **21,802,784** (~21.80M) |
| EfficientNetB0 | **5,330,571** (~5.33M) | **4,049,571** (~4.05M) |

Ratio (full ImageNet heads): 23.85M / 5.33M ≈ **4.5×** fewer parameters for EfficientNet-B0.

## What is intentionally *not* changed in M4

No new losses, no ensemble, no Grad-CAM (that can wait for later analysis), no dataset remix, no clinical deployment claims. One backbone swap — then measure.
