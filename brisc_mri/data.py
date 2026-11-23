"""Portable manifest loading and matching train/inference preprocessing."""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

CLASS_NAMES = ('glioma', 'meningioma', 'pituitary', 'no_tumor')
PLANE_CODES = ('ax', 'co', 'sa')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(image, size=256):
    image = image.convert('L').resize((size, size), Image.Resampling.BILINEAR)
    return TF.normalize(TF.to_tensor(image).repeat(3, 1, 1), MEAN, STD)


def resolve_path(value, root):
    """Rebase historical Kaggle paths at their task directory."""
    path = str(value).replace('\\', '/')
    for task in ('classification_task/', 'segmentation_task/'):
        if task in path:
            return str(Path(root) / path[path.index(task):])
    return str(Path(root) / path)


def load_manifest(path, data_root=None, check_files=True):
    df = pd.read_csv(path, keep_default_na=False)
    required = {'image_path', 'mask_path', 'tumor_label', 'plane_code',
                'pseudo_patient_id', 'strict_protocol'}
    if missing := required - set(df.columns):
        raise ValueError(f'Manifest is missing columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Manifest is empty.')
    if not df.tumor_label.isin(CLASS_NAMES).all() or not df.plane_code.isin(PLANE_CODES).all():
        raise ValueError('Manifest contains unknown class or plane labels.')
    if not df.strict_protocol.isin(('train', 'val_cal', 'test')).all():
        raise ValueError('Unknown strict_protocol split.')
    if df.groupby('pseudo_patient_id').strict_protocol.nunique().gt(1).any():
        raise ValueError('Pseudo-patient groups overlap across strict splits.')
    if data_root is not None:
        for column in ('image_path', 'mask_path'):
            df[column] = df[column].map(lambda p: resolve_path(p, data_root) if p else '')
    df['has_mask'] = df.mask_path.ne('')
    if (df.tumor_label.ne('no_tumor') & ~df.has_mask).any():
        raise ValueError('Tumor-positive scans require ground-truth masks.')
    if check_files:
        paths = list(df.image_path) + list(df.loc[df.has_mask, 'mask_path'])
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} dataset files missing; first: {missing[0]}. Set --data-root to the directory containing classification_task and segmentation_task.')
    return df


def development_splits(df, seed=42):
    """Reserve calibration groups before fitting; never use them for selection."""
    from sklearn.model_selection import GroupShuffleSplit
    training = df[df.strict_protocol.eq('train')].copy()
    pool = df[df.strict_protocol.eq('val_cal')].copy()
    test = df[df.strict_protocol.eq('test')].copy()
    if training.empty or test.empty or pool.pseudo_patient_id.nunique() < 2:
        raise ValueError('Need train, test and at least two val_cal groups.')
    val_idx, cal_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(pool, groups=pool.pseudo_patient_id))
    return training, pool.iloc[val_idx].copy(), pool.iloc[cal_idx].copy(), test


class BRISCMultiTaskDataset(Dataset):
    def __init__(self, df, img_size=256, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row.image_path) as source:
            img = source.convert('L').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        if row.tumor_label == 'no_tumor':
            mask = Image.new('L', img.size, 0)
        else:
            with Image.open(row.mask_path) as source:
                mask = source.convert('L').resize(img.size, Image.Resampling.NEAREST)
        if self.is_train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)
        return (preprocess(img, self.img_size), (TF.to_tensor(mask) > 0.5).float(),
                torch.tensor(CLASS_NAMES.index(row.tumor_label)),
                torch.tensor(PLANE_CODES.index(row.plane_code)))
