"""Local CPU inference script for the BRISC 2025 multi-task MRI model.

Usage
-----
    python3 predict.py <path_to_mri_image.jpg> [axial|coronal|sagittal]

The script:
  1. Loads ``models/brisc2025_bundle.joblib`` (CPU-ready compressed bundle).
  2. Rebuilds the PlaneConditionedMultiTaskUNet from the saved state_dict.
  3. Preprocesses the given MRI slice (resize → normalise → tensor).
  4. Outputs the top tumour-class prediction and conformal prediction set.
  5. Saves a side-by-side segmentation overlay PNG.

Requirements
------------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install joblib numpy pillow matplotlib
"""

from __future__ import annotations

import sys
import pathlib
import joblib
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from src.model import PlaneConditionedMultiTaskUNet  # exact class matching state_dict keys

# ── Default paths ────────────────────────────────────────────────────────────
_BUNDLE_PATH = pathlib.Path("models/brisc2025_bundle.joblib")


def load_brisc_bundle(
    bundle_path: str | pathlib.Path = _BUNDLE_PATH,
) -> tuple[PlaneConditionedMultiTaskUNet, dict]:
    """Load the joblib bundle and reconstruct the model on CPU.

    Args:
        bundle_path: Path to ``brisc2025_bundle.joblib``.

    Returns:
        ``(model, bundle)`` where model is in eval mode on CPU.
    """
    bundle_path = pathlib.Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle not found at '{bundle_path}'.\n"
            "Download it from Google Drive and place it in the models/ directory.\n"
            "See README.md → Step 2 for instructions."
        )

    print(f"Loading bundle from {bundle_path} …", flush=True)
    bundle: dict = joblib.load(bundle_path)

    model = PlaneConditionedMultiTaskUNet(
        num_classes=len(bundle["classes"]),
        num_planes=len(bundle["planes"]),
        pretrained=False,  # weights come from the bundle
    )
    model.load_state_dict(bundle["state_dict"])
    model.to("cpu")
    model.eval()
    print("Model loaded successfully on CPU.", flush=True)
    return model, bundle


def preprocess_mri(image_path: str | pathlib.Path, bundle: dict) -> torch.Tensor:
    """Open an MRI image file and return a normalised ``[1, 3, H, W]`` tensor.

    Args:
        image_path: Path to any image format supported by Pillow.
        bundle:     The loaded joblib bundle (supplies img_size / norm stats).

    Returns:
        Float32 tensor of shape ``[1, 3, img_size, img_size]``.
    """
    size: int = bundle["img_size"]
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(bundle["norm_mean"], dtype=np.float32)
    std = np.array(bundle["norm_std"], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC → CHW → add batch dim
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, 256, 256]


def predict_slice(
    image_path: str | pathlib.Path,
    plane: str = "axial",
    out_vis: str = "prediction_overlay.png",
    bundle_path: str | pathlib.Path = _BUNDLE_PATH,
) -> dict:
    """Run inference on a single MRI slice and save a segmentation overlay.

    Args:
        image_path:  Path to the input MRI image.
        plane:       Anatomical plane – ``"axial"``, ``"coronal"``, or
                     ``"sagittal"``.
        out_vis:     Output path for the PNG overlay figure.
        bundle_path: Path to ``brisc2025_bundle.joblib``.

    Returns:
        Dict with keys ``pred_class``, ``probs``, ``conformal_set``,
        ``mask_prob``, and ``overlay_path``.
    """
    torch.set_num_threads(4)  # limit CPU parallelism for laptop friendliness

    model, bundle = load_brisc_bundle(bundle_path)

    classes: list[str] = bundle["classes"]
    planes: list[str] = bundle["planes"]
    plane_map = {p: i for i, p in enumerate(planes)}

    if plane.lower() not in plane_map:
        raise ValueError(
            f"plane='{plane}' is not recognised.  Choose from: {planes}"
        )

    img_tensor = preprocess_mri(image_path, bundle)
    plane_tensor = torch.tensor([plane_map[plane.lower()]], dtype=torch.long)

    with torch.inference_mode():
        cls_logits, mask_logits = model(img_tensor, plane_tensor)
        probs: np.ndarray = torch.softmax(cls_logits, dim=1)[0].numpy()
        mask_prob: np.ndarray = torch.sigmoid(mask_logits)[0, 0].numpy()

    pred_idx = int(np.argmax(probs))
    q_hat: float = bundle.get("conformal_qhat", 0.90)
    conformal_set = [classes[i] for i, p in enumerate(probs) if (1.0 - p) <= q_hat]

    # ── Console output ───────────────────────────────────────────────────────
    print()
    print(f"Anatomical Plane  : {plane.capitalize()}")
    print(f"Top Prediction    : {classes[pred_idx]}  ({probs[pred_idx] * 100:.2f}%)")
    print(f"Conformal Set     : {conformal_set}  (q̂ = {q_hat:.3f})")
    print("─" * 44)
    for cls_name, p in zip(classes, probs):
        bar = "█" * int(p * 20)
        print(f"  {cls_name:14s}: {p * 100:5.2f}%  {bar}")
    print()

    # ── Segmentation overlay figure ──────────────────────────────────────────
    size: int = bundle["img_size"]
    orig = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(orig)
    axes[0].set_title(f"Input MRI  ·  {plane.capitalize()} plane", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(orig)
    axes[1].imshow(mask_prob > 0.5, cmap="autumn", alpha=0.45)
    axes[1].set_title(
        f"Pred: {classes[pred_idx]}  ({probs[pred_idx] * 100:.1f}%)\n"
        f"Conformal set: {conformal_set}",
        fontsize=10,
    )
    axes[1].axis("off")

    plt.tight_layout()
    out_path = pathlib.Path(out_vis)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved segmentation overlay → {out_path.resolve()}")

    return {
        "pred_class": classes[pred_idx],
        "probs": dict(zip(classes, probs.tolist())),
        "conformal_set": conformal_set,
        "mask_prob": mask_prob,
        "overlay_path": str(out_path.resolve()),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python3 predict.py <path_to_mri_image.jpg> [axial|coronal|sagittal]")
        sys.exit(1)

    img_file = sys.argv[1]
    scan_plane = sys.argv[2] if len(sys.argv) > 2 else "axial"
    predict_slice(img_file, scan_plane)
