"""BRISC 2025 · Local MRI inference interface.

Supports both backends:
  • models/brisc2025_bundle.joblib  (CPU-ready compressed bundle — preferred)
  • checkpoints/best_brisc_model.pt (original Colab checkpoint)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from matplotlib.figure import Figure

# ── Page config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="BRISC 2025 · MRI Inference",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    .stApp {background:#0b1220; color:#e5edf8;}
    [data-testid="stSidebar"] {background:#111c2e;}
    .stButton button {border-radius:8px;}
    .metric-label {font-size:0.8rem !important;}
    </style>""",
    unsafe_allow_html=True,
)

# ── Architecture (must match the saved state_dict exactly) ───────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PlaneConditionedMultiTaskUNet(nn.Module):
    def __init__(self, num_classes=4, num_planes=3, pretrained=False):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        self.plane_emb = nn.Embedding(num_planes, 64)
        self.film = nn.Sequential(nn.Linear(64, 256), nn.ReLU(inplace=True), nn.Linear(256, 512 * 2))
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_up = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, x, plane_idx):
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)
        p_vec = self.plane_emb(plane_idx)
        gamma, beta = self.film(p_vec).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        cond_feat = bottleneck * (1.0 + gamma) + beta
        cls_logits = self.cls_head(cond_feat)
        d4 = self.dec4(cond_feat, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        return cls_logits, self.final_up(d1)


# ── Model loading (cached per session) ───────────────────────────────────────
BUNDLE_PATH = ROOT / "models" / "brisc2025_bundle.joblib"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_brisc_model.pt"

FALLBACK_CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")
FALLBACK_PLANES  = ("axial", "coronal", "sagittal")
PLANE_CODE_MAP   = {"axial": "ax", "coronal": "co", "sagittal": "sa"}

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(source_path: str):
    """Load model from joblib bundle or raw .pt checkpoint. Cached for the session."""
    path = Path(source_path)
    q_hat = None
    classes = list(FALLBACK_CLASSES)
    planes  = list(FALLBACK_PLANES)
    img_size = 256
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    if path.suffix == ".joblib":
        bundle = joblib.load(path)
        state_dict = bundle["state_dict"]
        q_hat    = bundle.get("conformal_qhat")
        classes  = bundle.get("classes", classes)
        planes   = bundle.get("planes", planes)
        img_size = bundle.get("img_size", img_size)
        norm_mean = bundle.get("norm_mean", norm_mean)
        norm_std  = bundle.get("norm_std", norm_std)
    else:
        from collections.abc import Mapping
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
            q_hat = payload.get("q_hat")
        else:
            state_dict = payload

    model = PlaneConditionedMultiTaskUNet(
        num_classes=len(classes), num_planes=len(planes), pretrained=False
    )
    model.load_state_dict(state_dict, strict=True)
    model.to("cpu").eval()
    torch.set_num_threads(4)

    return model, {
        "q_hat": float(q_hat) if (q_hat is not None and math.isfinite(float(q_hat))) else None,
        "classes": classes,
        "planes": planes,
        "img_size": img_size,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def preprocess(image: Image.Image, meta: dict) -> torch.Tensor:
    size = meta["img_size"]
    img  = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    arr  = (arr - np.array(meta["norm_mean"])) / np.array(meta["norm_std"])
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def run_inference(model, meta: dict, image: Image.Image, plane: str, n_passes: int):
    """Single deterministic classification + MC-dropout segmentation + Grad-CAM."""
    plane_idx = meta["planes"].index(plane)
    x = preprocess(image, meta).requires_grad_(True)
    p = torch.tensor([plane_idx], dtype=torch.long)

    # Grad-CAM
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []
    fwd = model.enc4.register_forward_hook(lambda _, __, o: activations.append(o))
    bwd = model.enc4.register_full_backward_hook(lambda _, __, g: gradients.append(g[0]))

    cls_logits, _ = model(x, p)
    probs = torch.softmax(cls_logits, dim=1)[0].detach().numpy()
    target = int(probs.argmax())
    model.zero_grad()
    cls_logits[0, target].backward()
    fwd.remove(); bwd.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(meta["img_size"], meta["img_size"]), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # MC-dropout uncertainty
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()
    seg_samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            _, seg_out = model(x.detach(), p)
            seg_samples.append(torch.sigmoid(seg_out)[0, 0].numpy())
    model.eval()

    mean_mask = np.mean(seg_samples, axis=0)
    uncertainty = np.std(seg_samples, axis=0)

    return probs, target, mean_mask, uncertainty, cam


def render_figure(image: Image.Image, mean_mask, uncertainty, cam, meta: dict, pred_label: str) -> bytes:
    """Return a 3-panel visualization PNG as bytes."""
    size = meta["img_size"]
    display = image.convert("L").resize((size, size), Image.BILINEAR)
    fig = Figure(figsize=(13, 4), facecolor="#101827")
    axes = fig.subplots(1, 3)

    # Panel 1: MRI + predicted mask
    axes[0].imshow(display, cmap="gray")
    axes[0].imshow(np.ma.masked_where(mean_mask <= 0.5, mean_mask), cmap="Reds", vmin=0, vmax=1, alpha=0.65)
    axes[0].set_title("Predicted tumor mask", color="white", fontsize=11)

    # Panel 2: MC-dropout uncertainty heatmap
    uc = axes[1].imshow(uncertainty, cmap="magma", vmin=0, vmax=max(0.01, float(uncertainty.max())))
    fig.colorbar(uc, ax=axes[1], fraction=0.046).ax.tick_params(colors="white")
    axes[1].set_title("MC-Dropout uncertainty (σ)", color="white", fontsize=11)

    # Panel 3: Grad-CAM
    axes[2].imshow(display, cmap="gray")
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(f"Grad-CAM · {pred_label}", color="white", fontsize=11)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Auto-detect best available source
    default_source = str(BUNDLE_PATH) if BUNDLE_PATH.exists() else str(CHECKPOINT_PATH)
    model_source = st.text_input("Model source (bundle or .pt)", value=default_source)

    plane_name = st.selectbox(
        "Anatomical plane",
        ["Axial", "Coronal", "Sagittal"],
        index=0,
    )
    n_passes = st.slider("MC-dropout passes", min_value=1, max_value=30, value=10,
                         help="More passes = steadier uncertainty, slower inference.")

    st.divider()
    if BUNDLE_PATH.exists():
        sz = BUNDLE_PATH.stat().st_size / 1e6
        st.success(f"✅ Bundle found ({sz:.0f} MB)")
    else:
        st.warning("⚠️ Bundle not found at models/brisc2025_bundle.joblib")

    st.markdown(
        "**Expected input**\n\nPNG or JPEG T1 MRI slice.  "
        "DICOM / NIfTI volumes must be exported as JPEG/PNG first.\n\n"
        "Uploads are processed in memory and are not saved to disk."
    )
    st.caption("Research demo — not validated for clinical diagnosis.")


# ── Header ────────────────────────────────────────────────────────────────────
st.caption("BRISC 2025 / MULTI-TASK MRI")
st.title("🧠 Brain MRI Research Interface")
st.write(
    "Drag and drop (or click to upload) a T1 MRI slice below.  "
    "The model predicts the **tumour class**, draws the **segmentation mask**, "
    "shows **pixel-level uncertainty** (MC-Dropout), and highlights the "
    "regions driving the decision (**Grad-CAM**)."
)

scan_tab, about_tab = st.tabs(["🔬 Analyze a slice", "📋 About the model"])

# ── About tab ─────────────────────────────────────────────────────────────────
with about_tab:
    st.subheader("Plane-conditioned multi-task U-Net")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Architecture**
- ResNet34 encoder (pretrained on ImageNet)
- FiLM conditioning at bottleneck (512-ch)
- 4-class tumour classifier head
- U-Net segmentation decoder

**Training** (strict leak-free split)
- Train: 3,599 · Val/Cal: 1,200 · Test: 1,201
        """)
    with col2:
        st.markdown("""
**Test-set results (1,201 unseen scans)**
| Metric | Score |
|---|---|
| Accuracy | **99.58 %** |
| Macro-F1 | **99.58 %** |
| Tumour Dice | **0.840** |
| Tumour IoU | **0.757** |
| Conformal coverage | **95.67 %** |
        """)
    diag = ROOT / "assets" / "brisc2025_clinical_diagnostics.png"
    if diag.exists():
        st.image(str(diag), caption="Training diagnostic figure — not an uploaded-scan prediction.")

# ── Scan tab ──────────────────────────────────────────────────────────────────
with scan_tab:
    model_path = Path(model_source).expanduser()
    model_ready = model_path.exists() and model_path.is_file()
    if not model_ready:
        st.error(
            f"Model not found at `{model_source}`.  "
            "Make sure `models/brisc2025_bundle.joblib` exists (see README)."
        )

    # ── Upload widget ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload an MRI slice",
        type=["png", "jpg", "jpeg"],
        help="Drag and drop or click to browse. Max 20 MB.",
        label_visibility="collapsed",
    )

    image: Image.Image | None = None
    content: bytes | None = None

    if uploaded is not None:
        try:
            content = uploaded.getvalue()
            if len(content) > 20 * 1024 * 1024:
                st.error("File too large — please use an image under 20 MB.")
            else:
                with Image.open(BytesIO(content)) as src:
                    if src.width * src.height > 20_000_000:
                        st.error("Image too large — please use fewer than 20 megapixels.")
                    else:
                        image = src.convert("L").copy()
                prev_col, _ = st.columns([1, 2])
                if image:
                    prev_col.image(image, caption=f"Uploaded: {uploaded.name}", width=280)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            st.error(f"Could not open image: {e}")

    # ── Run button ────────────────────────────────────────────────────────────
    sig = (str(model_path.resolve()), model_path.stat().st_mtime_ns) if model_ready else None
    cache_key = (hashlib.sha256(content).hexdigest() if content else None, sig, plane_name, n_passes)

    if st.session_state.get("cache_key") != cache_key:
        st.session_state.pop("result", None)
    st.session_state["cache_key"] = cache_key

    analyze_btn = st.button(
        "▶ Analyze MRI slice",
        type="primary",
        disabled=not (model_ready and image is not None),
    )

    if analyze_btn:
        try:
            model, meta = load_model(str(model_path))
            plane_key = plane_name.lower()   # "axial" / "coronal" / "sagittal"

            with st.spinner(f"Running inference on {plane_name} plane…"):
                probs, pred_idx, mean_mask, uncertainty, cam = run_inference(
                    model, meta, image, plane_key, n_passes
                )
                pred_label = meta["classes"][pred_idx]
                figure_png = render_figure(image, mean_mask, uncertainty, cam, meta, pred_label)

            q_hat = meta["q_hat"]
            conformal_set = (
                [meta["classes"][i] for i, p in enumerate(probs) if (1.0 - p) <= q_hat]
                if q_hat is not None else None
            )
            st.session_state["result"] = {
                "probs": probs,
                "pred_idx": pred_idx,
                "pred_label": pred_label,
                "mean_mask": mean_mask,
                "figure_png": figure_png,
                "q_hat": q_hat,
                "conformal_set": conformal_set,
                "classes": meta["classes"],
            }
        except Exception as e:
            st.error(f"Inference failed: {e}")
            raise

    # ── Results ───────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        r = st.session_state["result"]
        st.divider()

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted class", r["pred_label"].replace("_", " ").title())
        m2.metric("Confidence", f"{r['probs'][r['pred_idx']]*100:.1f} %")

        q_hat = r["q_hat"]
        if q_hat is not None:
            cset = r["conformal_set"]
            cset_str = ", ".join(c.replace("_", " ").title() for c in cset) if cset else "∅ (empty)"
            m3.metric("Conformal set", cset_str)
            m4.metric("q̂  (95 % coverage)", f"{q_hat:.4f}")
            if len(cset) != 1:
                st.warning(
                    f"⚠️ **Review required** — conformal set has {len(cset)} class(es): {cset_str}.  "
                    "The model is uncertain; a radiologist should review."
                )
            else:
                st.success(f"✅ Single-class conformal set → **{cset_str}** at 95 % coverage guarantee.")
        else:
            m3.metric("Conformal set", "N/A (no calibration)")
            m4.metric("q̂", "—")

        # Probability bar chart
        st.subheader("Class probabilities")
        prob_dict = {c.replace("_", " ").title(): float(p) for c, p in zip(r["classes"], r["probs"])}
        st.bar_chart(prob_dict, color="#4f8ef7")

        # Visualization panels
        st.subheader("Segmentation · Uncertainty · Grad-CAM")
        st.image(r["figure_png"], use_container_width=True)
        st.caption(
            "Left: predicted tumour mask (threshold 0.5)  ·  "
            "Centre: MC-Dropout pixel std across passes  ·  "
            "Right: Grad-CAM highlighting decisive regions"
        )

        # Download buttons
        st.divider()
        dl1, dl2, dl3 = st.columns(3)
        report = {
            "predicted_class": r["pred_label"],
            "confidence": float(r["probs"][r["pred_idx"]]),
            "class_probabilities": {c: float(p) for c, p in zip(r["classes"], r["probs"])},
            "conformal_set": r["conformal_set"],
            "q_hat": r["q_hat"],
            "plane": plane_name,
            "mc_passes": n_passes,
            "research_only": True,
        }
        dl1.download_button("⬇ Report (JSON)", json.dumps(report, indent=2),
                            "brisc_report.json", "application/json")
        mask_buf = BytesIO()
        Image.fromarray((r["mean_mask"] > 0.5).astype(np.uint8) * 255).save(mask_buf, format="PNG")
        dl2.download_button("⬇ Mask (PNG)", mask_buf.getvalue(), "brisc_mask.png", "image/png")
        dl3.download_button("⬇ Visualization (PNG)", r["figure_png"], "brisc_viz.png", "image/png")
