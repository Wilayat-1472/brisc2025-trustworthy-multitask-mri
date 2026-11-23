"""Find/download BRISC, validate the committed split, and report local paths."""
import argparse
from pathlib import Path
from .data import load_manifest, development_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', help='Directory containing classification_task and segmentation_task.')
    parser.add_argument('--manifest', default='data/brisc2025_clean.csv')
    args = parser.parse_args()
    if args.data_root:
        root = Path(args.data_root).expanduser()
    else:
        import kagglehub
        root = Path(kagglehub.dataset_download('briscdataset/brisc2025'))
        if not (root / 'classification_task').is_dir():
            candidates = list(root.rglob('classification_task'))
            if len(candidates) != 1:
                raise FileNotFoundError('Could not identify dataset root; supply --data-root.')
            root = candidates[0].parent
    df = load_manifest(args.manifest, root)
    parts = development_splits(df)
    print(f'Data root: {root.resolve()}')
    print(dict(zip(('train', 'validation', 'calibration', 'test'), map(len, parts))))
    print('File existence and pseudo-patient split checks passed. pHash groups are not verified patient identities.')


if __name__ == '__main__':
    main()
