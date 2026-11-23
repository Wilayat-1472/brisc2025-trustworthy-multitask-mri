# Historical notebook results

The following table is preserved from the original README. These are author-reported
results from the original notebook, not independently reproduced results of the
refactored pipeline. The raw downloaded checkpoint has no calibration metadata.
The original validation set was also used for model selection and calibration,
so a standard split-conformal coverage guarantee does not follow from that run.
The revised pipeline uses separate validation and calibration groups; rerun it to
obtain metrics under that protocol.

pHash grouping detects near-duplicate images. It does not establish patient
identity or prove that all patient-level leakage has been eliminated. The metric
previously called healthy brain pixel specificity measures the fraction of healthy
scans with at most ten predicted foreground pixels, not pixel-level specificity.

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
