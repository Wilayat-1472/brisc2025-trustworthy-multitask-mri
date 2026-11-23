"""Evaluate a frozen checkpoint on the manifest's test split."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
import torch
from .data import load_manifest, BRISCMultiTaskDataset, CLASS_NAMES
from .inference import Predictor, prediction_set
from .train import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/best_brisc_model.pt')
    parser.add_argument('--manifest', default='data/brisc2025_clean.csv')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='reports/test_metrics.json')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, 'cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    if 'manifest_sha256' in payload and payload['manifest_sha256'] != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
        parser.error('Manifest differs from the one used for training; use the original manifest with --data-root.')
    df = load_manifest(args.manifest, args.data_root)
    test = df[df.strict_protocol.eq('test')]
    if test.empty:
        parser.error('Test split is empty.')
    probs, labels, dices, ious, fps = collect(predictor.model, DataLoader(BRISCMultiTaskDataset(test), batch_size=args.batch_size), predictor.device)
    result = {'test_samples': len(labels), 'accuracy': float(accuracy_score(labels, probs.argmax(1))),
              'macro_f1': float(f1_score(labels, probs.argmax(1), labels=range(4), average='macro', zero_division=0)),
              'tumor_dice': float(np.mean(dices)) if dices else None,
              'tumor_iou': float(np.mean(ious)) if ious else None,
              'healthy_scans': len(fps), 'healthy_scans_over_10_fp_pixels': sum(x > 10 for x in fps),
              'classification_report': classification_report(labels, probs.argmax(1), labels=range(4), target_names=CLASS_NAMES, output_dict=True, zero_division=0)}
    if predictor.q_hat is not None:
        sets = [prediction_set(p, predictor.q_hat) for p in probs]
        result['conformal'] = {'alpha': predictor.alpha, 'q_hat': predictor.q_hat,
                              'empirical_coverage': float(np.mean([CLASS_NAMES[y] in s for y, s in zip(labels, sets)])),
                              'average_set_size': float(np.mean([len(s) for s in sets])),
                              'review_required': sum(len(s) != 1 for s in sets)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
