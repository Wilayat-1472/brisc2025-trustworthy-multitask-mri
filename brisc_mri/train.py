"""Train with group-disjoint selection and calibration sets: python -m brisc_mri.train."""
import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .data import CLASS_NAMES, PLANE_CODES, BRISCMultiTaskDataset, load_manifest, development_splits
from .model import PlaneConditionedMultiTaskUNet
from .losses import dice_bce_loss, compute_tumor_dice
from .inference import conformal_threshold


def collect(model, loader, device):
    model.eval()
    probabilities, labels, dices, ious, healthy_fps = [], [], [], [], []
    with torch.no_grad():
        for images, masks, targets, planes in loader:
            images, masks = images.to(device), masks.to(device)
            targets, planes = targets.to(device), planes.to(device)
            cls, seg = model(images, planes)
            probabilities.extend(cls.softmax(1).cpu().tolist())
            labels.extend(targets.cpu().tolist())
            dices.extend(compute_tumor_dice(seg, masks, targets))
            binary = seg.sigmoid() > 0.5
            for prediction, mask, label in zip(binary, masks > 0.5, targets):
                if label.item() == 3:
                    healthy_fps.append(int(prediction.sum().item()))
                else:
                    intersection = (prediction & mask).sum().item()
                    union = (prediction | mask).sum().item()
                    ious.append((intersection + 1e-5) / (union + 1e-5))
    return np.asarray(probabilities), np.asarray(labels), dices, ious, healthy_fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/brisc2025_clean.csv')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='checkpoints/best_brisc_model.pt')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--pretrained', action='store_true', help='Download ImageNet encoder weights for training.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.alpha < 1:
        parser.error('epochs/batch-size must be positive and 0 < alpha < 1')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_manifest(args.manifest, args.data_root)
    train, val, calibration, _ = development_splits(df, args.seed)
    loaders = [DataLoader(BRISCMultiTaskDataset(part, is_train=(i == 0)), batch_size=args.batch_size,
                          shuffle=(i == 0), num_workers=args.workers) for i, part in enumerate((train, val, calibration))]
    print(f'Device: {device}; train={len(train)}, validation={len(val)}, calibration={len(calibration)}', flush=True)
    model = PlaneConditionedMultiTaskUNet(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -float('inf')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'class_names': list(CLASS_NAMES), 'plane_codes': list(PLANE_CODES),
                  'seed': args.seed, 'image_size': 256,
                  'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  'calibration_group_ids': calibration.pseudo_patient_id.astype(int).tolist(),
                  'validation_group_ids': val.pseudo_patient_id.astype(int).tolist()}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks, labels, planes in loaders[0]:
            images, masks, labels, planes = [tensor.to(device) for tensor in (images, masks, labels, planes)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
                cls, seg = model(images, planes)
                loss = criterion(cls, labels) + 1.5 * dice_bce_loss(seg, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
        scheduler.step()
        probs, labels, dices, _, _ = collect(model, loaders[1], device)
        macro_f1 = f1_score(labels, probs.argmax(1), labels=range(4), average='macro', zero_division=0)
        dice = float(np.mean(dices)) if dices else 0.0
        score = float((macro_f1 + dice) / 2)
        if score > best:
            best = score
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'validation_score': score,
                        **provenance}, output)
        print(f'Epoch {epoch}/{args.epochs}: loss={total_loss/len(train):.4f}, val F1={macro_f1:.4f}, val Dice={dice:.4f}', flush=True)
    payload = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(payload['model_state_dict'])
    probs, labels, _, _, _ = collect(model, loaders[2], device)
    payload.update(q_hat=conformal_threshold(probs, labels, args.alpha), alpha=args.alpha,
                   calibration_size=len(labels), calibration_protocol='held-out-groups')
    torch.save(payload, output)
    print(f'Saved calibrated checkpoint: {output}', flush=True)


if __name__ == '__main__':
    main()
