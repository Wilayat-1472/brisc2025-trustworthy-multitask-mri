"""Checkpoint loading, deterministic classification, MC masks and Grad-CAM."""
from pathlib import Path
from collections.abc import Mapping
import math
from threading import RLock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import CLASS_NAMES, PLANE_CODES, preprocess
from .model import PlaneConditionedMultiTaskUNet


def conformal_threshold(probs, labels, alpha=0.05):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    if not 0 < alpha < 1 or labels.ndim != 1 or labels.size == 0:
        raise ValueError('Calibration requires samples and 0 < alpha < 1.')
    if probs.shape != (labels.size, len(CLASS_NAMES)):
        raise ValueError('Calibration needs one four-class probability row per label.')
    if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError('Calibration probabilities must be finite and sum to one.')
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError('Calibration labels must be integer class indices 0-3.')
    labels = labels.astype(int)
    scores = 1.0 - probs[np.arange(len(labels)), labels]
    rank = math.ceil((len(labels) + 1) * (1 - alpha))
    # With too few calibration samples, the finite-sample set is all classes.
    return 1.0 if rank > len(scores) else float(np.sort(scores)[rank - 1])


def prediction_set(probs, q_hat):
    if q_hat is None:
        return None
    return [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= 1.0 - q_hat]


class Predictor:
    def __init__(self, checkpoint, device='cpu'):
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {path}. Copy best_brisc_model.pt from Colab into checkpoints/.')
        self.device = torch.device(device)
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("Checkpoint must contain a state dictionary or a model_state_dict mapping.")
        self.q_hat = None
        self.alpha = None
        if 'model_state_dict' in payload:
            state = payload['model_state_dict']
            self.q_hat = payload.get('q_hat')
            self.alpha = payload.get('alpha')
            if tuple(payload.get('class_names', CLASS_NAMES)) != CLASS_NAMES:
                raise ValueError('Checkpoint class order does not match the model.')
            if tuple(payload.get('plane_codes', PLANE_CODES)) != PLANE_CODES:
                raise ValueError('Checkpoint plane order does not match the model.')
        else:
            state = payload  # Original notebook torch.save(model.state_dict(), ...).
        if self.q_hat is not None and (not math.isfinite(float(self.q_hat)) or not 0 <= self.q_hat <= 1):
            raise ValueError('Invalid calibration threshold in checkpoint.')
        self.model = PlaneConditionedMultiTaskUNet(pretrained=False).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.lock = RLock()

    def predict(self, image, plane='ax', n_passes=10):
        if plane not in PLANE_CODES or not 1 <= n_passes <= 50:
            raise ValueError('Choose ax/co/sa and 1–50 uncertainty passes.')
        with self.lock:
            return self._predict(image, plane, n_passes)

    def _predict(self, image, plane, n_passes):
        model = self.model
        model.eval()
        x = preprocess(image).unsqueeze(0).to(self.device).requires_grad_(True)
        p = torch.tensor([PLANE_CODES.index(plane)], device=self.device)
        activations = []
        hook = model.enc4.register_forward_hook(lambda _m, _i, out: activations.append(out))
        try:
            with torch.enable_grad():
                logits, _ = model(x, p)
                probabilities = logits.softmax(1)[0].detach().cpu().numpy()
                target = int(probabilities.argmax())
                gradients, = torch.autograd.grad(logits[0, target], activations[0])
                cam = F.relu((gradients.mean((2, 3), keepdim=True) * activations[0]).sum(1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
                cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        finally:
            hook.remove()
        masks = []
        try:
            # Classification stays deterministic and matches calibration. Only
            # segmentation is sampled; BatchNorm statistics never update.
            for module in model.modules():
                if isinstance(module, nn.Dropout2d):
                    module.train()
            with torch.no_grad():
                for _ in range(n_passes):
                    _, segmentation = model(x.detach(), p)
                    masks.append(segmentation.sigmoid()[0, 0].cpu().numpy())
        finally:
            model.eval()
        labels = prediction_set(probabilities, self.q_hat)
        return {'probabilities': probabilities, 'predicted_class': CLASS_NAMES[target],
                'prediction_set': labels, 'review_required': None if labels is None else len(labels) != 1,
                'mask': np.mean(masks, axis=0), 'uncertainty': np.std(masks, axis=0),
                'cam': cam, 'cam_target': CLASS_NAMES[target]}
