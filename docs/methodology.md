# Methodology and limitations

## Similarity-group separation

The committed manifest records 6,000 images grouped by 64-bit perceptual hash
connected components at Hamming distance ≤6. It contains 3,650 similarity groups.
The strict split contains 3,599 training, 1,200 val_cal and 1,201 test images.
The loader checks that no recorded group spans strict splits.

pHash cannot establish patient identity. Similar slices from one patient can
escape this threshold, and unrelated patients can share a similarity group.
Describe the split as **pHash-group-disjoint**, rather than verified patient-level
separation or proof of zero leakage. The refactor reuses the committed group
assignments; it does not recompute the original hash audit.

## Model selection and calibration

The original notebook selected the best epoch and calibrated on the same
val_cal set. The revised training command partitions that pool by group before
training. Model selection uses only the validation part. Calibration uses only
the calibration part after freezing the selected model. GroupShuffleSplit with
seed 42 makes this partition repeatable, but does not guarantee class balance.

Nonconformity scores are `1 - P(true_class)`. For n calibration samples, the
threshold is the ceil((n+1)(1-alpha))-th smallest score, with threshold 1 when
that rank exceeds n. Prediction sets contain classes whose deterministic
probability is at least `1 - q_hat`. Both empty and multi-class sets trigger
review. Sets remain unchanged by the number of MC segmentation passes.

Standard split-conformal marginal coverage relies on appropriate exchangeability
and a calibration set independent of model fitting/selection. Correlated slices,
unequal cluster sizes and deployment distribution shift can invalidate those
assumptions. Group disjointness alone does not establish them. A target coverage
of 95% is not a per-scan reliability or diagnostic accuracy guarantee.

## Uncertainty and attribution

Inference uses deterministic evaluation-mode classification, matching calibration.
Only decoder Dropout2d layers are enabled for repeated segmentation samples;
BatchNorm remains in evaluation mode. Pixel standard deviation reflects variation
under these dropout samples. It is not a validated pixel-error probability and
need not be concentrated only at tumor borders.

Grad-CAM targets the predicted class and uses the final encoder feature map.
Coarse saliency is not a tumor boundary or a causal explanation. Segmentation
and classification may disagree and are shown independently. All displayed maps
are resized to the 256 × 256 model grid.

## Original scores

Historical accuracy, Dice and coverage figures have not been reproduced during
this refactor. Full scientific validation requires the dataset, training setup,
checkpoint provenance and an appropriate held-out evaluation. The tests verify
software behavior and do not measure medical performance.
