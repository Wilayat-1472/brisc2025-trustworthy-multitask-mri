"""Training objective and tumor-only Dice metric."""
import torch
from torch import nn

def dice_bce_loss(seg_logits, targets, smooth=1e-5):
    bce = nn.functional.binary_cross_entropy_with_logits(seg_logits, targets)
    probs = torch.sigmoid(seg_logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())

def compute_tumor_dice(seg_logits, targets, labels, smooth=1e-5):
    """Computes Dice similarity specifically on tumor-positive scans (labels != 3)."""
    tumor_mask = (labels != 3)
    if tumor_mask.sum() == 0:
        return []
    preds = (torch.sigmoid(seg_logits[tumor_mask]) > 0.5).float()
    targs = targets[tumor_mask]
    inter = (preds * targs).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targs.sum(dim=(1, 2, 3))
    dices = ((2.0 * inter + smooth) / (union + smooth)).cpu().tolist()
    return dices
