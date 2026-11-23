"""In-memory result rendering; no shared temporary image files."""
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.figure import Figure


def render_result(image, result):
    fig = Figure(figsize=(12, 3.6), facecolor='#101827')
    axes = fig.subplots(1, 3)
    display = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    axes[0].imshow(display, cmap='gray')
    axes[0].imshow(np.ma.masked_where(result['mask'] <= 0.5, result['mask']), cmap='Reds', vmin=0, vmax=1, alpha=0.65)
    uncertainty = axes[1].imshow(result['uncertainty'], cmap='magma', vmin=0, vmax=max(0.01, float(result['uncertainty'].max())))
    fig.colorbar(uncertainty, ax=axes[1], fraction=0.046).ax.tick_params(colors='white')
    axes[2].imshow(display, cmap='gray')
    axes[2].imshow(result['cam'], cmap='jet', alpha=0.45, vmin=0, vmax=1)
    for ax, title in zip(axes, ('Predicted tumor mask', 'MC dropout · pixel standard deviation', f"Grad-CAM · {result['cam_target']}")):
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=130, facecolor=fig.get_facecolor())
    return buffer.getvalue()
