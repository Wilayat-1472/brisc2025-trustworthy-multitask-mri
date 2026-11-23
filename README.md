# Plane-Conditioned & Uncertainty-Aware Multi-Task Brain Tumor MRI Pipeline (BRISC 2025)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BRISC%202025-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/briscdataset/brisc2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, leak-free multi-task deep learning framework for simultaneous **4-class brain tumor diagnosis** (`Glioma`, `Meningioma`, `Pituitary`, `No Tumor`) and **pixel-level tumor segmentation** on T1-weighted MRI scans (`BRISC 2025`).

Unlike standard image classification benchmarks that evaluate on leaky slice splits and output overconfident point predictions, this pipeline enforces **perceptual-hash pseudo-patient clustering**, **anatomical plane conditioning**, and **post-hoc clinical uncertainty quantification**.

![Clinical Diagnostics](assets/brisc2025_clinical_diagnostics.png)

---

## Table of Contents
1. [Motivation & Key Contributions](#1-motivation--key-contributions)
2. [Data Leakage Audit & Split Protocol](#2-data-leakage-audit--split-protocol)
3. [Model Architecture](#3-model-architecture)
4. [Clinical Trustworthiness & Explainability](#4-clinical-trustworthiness--explainability)
5. [Experimental Results](#5-experimental-results)
6. [Repository Structure](#6-repository-structure)
7. [Quickstart & Reproduction](#7-quickstart--reproduction)
8. [Citation & Acknowledgments](#8-citation--acknowledgments)

---

## 1. Motivation & Key Contributions

Deploying automated neuro-oncology triage tools requires addressing three failure modes common in public MRI benchmarks:
* **Adjacent-Slice Data Leakage:** Multi-slice MRI datasets lacking patient identifiers frequently place adjacent slices from the same patient across both training and test splits, artificially inflating accuracy.
* **Anatomical Blindness:** Tumor morphology and spatial priors vary drastically across `Axial`, `Coronal`, and `Sagittal` viewing planes, yet standard CNNs process slices without orientation context.
* **Overconfident Predictions:** Standard softmax probabilities fail to abstain on ambiguous scans and provide no spatial uncertainty along blurry tumor borders.

### Core Contributions
1. **Perceptual-Hash (`pHash`) Leakage Audit:** Quantified adjacent-slice overlap across all 6,000 `BRISC 2025` scans and constructed a strictly leak-free, cluster-grouped benchmark split (`data/brisc2025_clean.csv`).
2. **Plane-Conditioned Multi-Task U-Net (`FiLM` Modulation):** Jointly trains a tumor classifier and a U-Net segmentation decoder from a shared `ResNet34` backbone modulated by slice orientation (`Axial`, `Coronal`, `Sagittal`).
3. **Zero-Mask Healthy Brain Regularization:** Unifies the 6,000-image classification task and 4,793-image segmentation task by assigning exact zero-masks to `No Tumor` scans, eliminating false-positive tumor hallucinations on healthy brains.
4. **Triple-Layer Clinical Safety Suite:** Combines **Split Conformal Prediction** (95% statistical coverage guarantee), **Monte Carlo Dropout Pixel Uncertainty** (boundary variance estimation), and **Grad-CAM vs. Mask Localization Auditing**.

---

## 2. Data Leakage Audit & Split Protocol

The raw `BRISC 2025` dataset contains **6,000 T1-weighted MRI slices** ($512 \times 512$) across four balanced classes (`1,401` Glioma, `1,635` Meningioma, `1,757` Pituitary, `1,207` No Tumor) and three anatomical planes (`Axial`, `Coronal`, `Sagittal`).

Before training, we computed 64-bit perceptual hashes (`imagehash.phash`) for all 6,000 slices and grouped near-identical/adjacent slices (Hamming distance $\le 6$) into connected components:

| Audit Metric | Finding |
| :--- | :--- |
| **Total MRI Slices Audited** | `6,000` |
| **Pseudo-Patient Clusters Identified** | `3,650` |
| **Clusters Spanning Official Train & Test** | `420` |
| **Official Test Scans with Near-Duplicate in Train** | **`559 / 1,000 (55.9%)`** |

To ensure rigorous generalization to unseen patients, all models are trained and evaluated on our **Strict Leak-Free Protocol** (`StratifiedGroupKFold` grouped by `pseudo_patient_id` and stratified by `tumor_class × anatomical_plane`):
* **Training Set:** `3,599` scans (synchronized random horizontal flips & $\pm 15^\circ$ rotations applied jointly to images and masks)
* **Validation / Conformal Calibration Set:** `1,200` scans
* **Held-Out Test Set:** `1,201` scans (zero pseudo-patient overlap with train/val)

---

## 3. Model Architecture

The network (`24.63M` parameters) unifies classification and dense segmentation into a single forward pass:

```text
Input MRI [B, 3, 256, 256] ──► ResNet34 Encoder ──► Bottleneck [B, 512, 8, 8]
                                                           │
Anatomical Plane [ax/co/sa] ──► Embedding + MLP ──► FiLM (γ, β) Modulation
                                                           │
                                            ┌──────────────┴──────────────┐
                                            ▼                             ▼
                                  Classification Head            U-Net Decoder (Skip-Fused)
                                  (GAP + Dropout 0.3)            (ConvTranspose + Dropout2d)
                                            │                             │
                                            ▼                             ▼
                                  4-Class Logits [B, 4]          Tumor Mask [B, 1, 256, 256]
```

### Feature-wise Linear Modulation (FiLM)
Embeds the slice plane index into learned channel-wise scale ($\gamma$) and shift ($\beta$) vectors applied at the $8 \times 8$ bottleneck:
$$F_{\text{cond}} = F \odot (1 + \gamma) + \beta$$
where $\gamma, \beta \in \mathbb{R}^{512}$ are generated via a 2-layer MLP (`Linear(64, 256) -> ReLU -> Linear(256, 1024)`) conditioned on the 64-dimensional learned plane embedding.

### Composite Multi-Task Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y; \epsilon = 0.05) + 1.5 \times \left(0.5 \mathcal{L}_{\text{BCE}}(\hat{M}, M) + 0.5 \mathcal{L}_{\text{SoftDice}}(\hat{M}, M)\right)$$
* **Classification Loss ($\mathcal{L}_{\text{CE}}$):** Label-smoothed cross-entropy ($\epsilon = 0.05$) to prevent logit overconfidence.
* **Segmentation Loss ($\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{SoftDice}}$):** Fuses pixel-level binary cross-entropy with global soft Dice loss, weighted at $1.5\times$ to enforce clean tumor boundary delineation.
* **Healthy Brain Regularization:** Slices with label `no_tumor` are paired with all-zero ground-truth masks ($M = \mathbf{0}$), penalizing false-positive tumor hallucinations.

---

## 4. Clinical Trustworthiness & Explainability

### Split Conformal Prediction (Selective Abstention)
* Non-conformity scores $s_i = 1 - \hat{P}(y_i \mid x_i)$ are calibrated on the 1,200-scan `val_cal` split at significance level $\alpha = 0.05$ (guaranteeing $\ge 95\%$ marginal coverage), yielding calibration threshold $\hat{q} = 0.0579$.
* Prediction sets for unseen test scans are defined as:
  $$C(x) = \{c : \hat{P}(c \mid x) \ge 1 - \hat{q} = 0.9421\}$$
* Ambiguous test scans whose top softmax probability falls below $1 - \hat{q}$ (0.9421) produce non-singleton sets ($|C(x)| \neq 1$) and are automatically flagged for radiologist review rather than forced into overconfident errors.
* On the 1,201 held-out test scans, conformal calibration achieved **95.67% empirical coverage** with an average prediction set size of **0.958 classes**.

### Monte Carlo (MC) Dropout Boundary Uncertainty
* At inference time, `Dropout2d(0.1)` layers within U-Net decoder blocks remain active across $T = 10$ stochastic forward passes while BatchNorm layers remain locked in evaluation mode.
* Pixel-wise standard deviation across passes maps epistemic spatial uncertainty:
  $$\sigma_{\text{pixel}}(u, v) = \sqrt{\frac{1}{T} \sum_{t=1}^T \left(\hat{M}_t(u, v) - \bar{M}(u, v)\right)^2}$$
* Uncertainty is concentrated exclusively along ambiguous, infiltrative tumor margins (Max Border Std: **0.138 – 0.237**), while tumor cores and healthy parenchyma remain invariant (**0.000 std**).

### Grad-CAM vs. Segmentation Mask Alignment Audit
* Comparing coarse $8 \times 8$ bottleneck Grad-CAM heatmaps against radiologist ground-truth masks revealed low spatial overlap (**CAM-Mask IoU: 0.004 – 0.078**), especially on small pituitary lesions, whereas the U-Net segmentation head achieved **0.8401 mean Dice** (**0.950** on pituitary cases).
* This empirically demonstrates why classifier saliency alone is insufficient for anatomical localization and confirms the clinical necessity of joint dense segmentation.

---

## 5. Experimental Results

### Held-Out Leak-Free Test Set (1,201 Unseen Scans)

| Evaluation Metric | Score | Notes |
| :--- | :--- | :--- |
| **Test Classification Accuracy** | **99.58%** | 1,196 / 1,201 correct on zero-leakage split |
| **Test Macro-F1 Score** | **0.9958** | Balanced across all 4 diagnostic classes |
| **Tumor Segmentation Dice (DSC)** | **0.8401** | Evaluated across all tumor-positive test scans |
| **Tumor Segmentation IoU** | **0.7573** | Intersection-over-Union with radiologist masks |
| **Healthy Brain Pixel Specificity** | **97.52%** | 236 / 242 `no_tumor` scans with $\le 10$ FP pixels |
| **Conformal Empirical Coverage** | **95.67%** | Target $\ge 95.0\%$ ($\hat{q} = 0.0579$, avg set size 0.958) |

### Per-Class Classification Breakdown

| Diagnostic Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 0.9964 | 0.9964 | 0.9964 | 280 |
| **Meningioma** | 0.9939 | 0.9939 | 0.9939 | 328 |
| **Pituitary** | 1.0000 | 0.9943 | 0.9971 | 351 |
| **No Tumor** | 0.9918 | 1.0000 | 0.9959 | 242 |
| **Macro Average** | **0.9955** | **0.9962** | **0.9958** | **1,201** |
| **Weighted Average** | **0.9959** | **0.9958** | **0.9958** | **1,201** |

---

## 6. Repository Structure

```text
brisc2025-trustworthy-multitask-mri/
├── assets/
│   └── brisc2025_clinical_diagnostics.png   # 5-panel MRI, GT, Mask, MC-Uncertainty & Grad-CAM visual
├── data/
│   └── brisc2025_clean.csv                # Unified 6,000-scan manifest with pseudo-patient IDs & splits
├── notebooks/
│   └── brisc2025_multitask_pipeline.ipynb # Step-by-step reproducible training & evaluation notebook
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. Quickstart & Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Wilayat-1472/brisc2025-trustworthy-multitask-mri.git
cd brisc2025-trustworthy-multitask-mri
pip install -r requirements.txt
```

### 2. Run in Google Colab or Local GPU
Open `notebooks/brisc2025_multitask_pipeline.ipynb` in Google Colab (select **Runtime → Change runtime type → T4 GPU**).

Execute cells sequentially to:
1. Fetch `briscdataset/brisc2025` directly via `kagglehub`.
2. Re-verify the perceptual-hash cluster audit and unified manifest (`data/brisc2025_clean.csv`).
3. Train `PlaneConditionedMultiTaskUNet` for 12 epochs (~6.5 minutes on a single T4 GPU using `torch.amp`).
4. Generate the test-set evaluation report, conformal calibration metrics, and the 5-panel clinical diagnostic figure.
5. (Optional) Save/load pre-trained model weights (`best_brisc_model.pt`) for immediate inference.

---

## 8. Citation & Acknowledgments

* **Dataset:** Brain Tumor MRI Dataset provided by the BRISC 2025 authors ([briscdataset/brisc2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)).
* **License:** This project is licensed under the [MIT License](LICENSE).
