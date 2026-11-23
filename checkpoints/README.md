# Model checkpoints

Copy the original Colab `/content/best_brisc_model.pt` here, or train with
`python -m brisc_mri.train`. Weights are excluded from Git.

The original raw state dictionary supports classification, masks, MC uncertainty,
and Grad-CAM. It has no calibration metadata, so the app does not invent a
threshold. Training through this package saves the state dictionary, class/plane
ordering, validation selection metadata, and a threshold computed from separate
held-out calibration groups.

For distribution, attach weights to a GitHub Release or a model hosting service.
Keep checkpoint download links, the training configuration and checksums together.
