"""
Member 2 - Ranjith (ML Engineer)
Task: MONAI Data Preprocessing Pipeline for BraTS 2021
"""
from pathlib import Path
from typing import List, Dict, Optional
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd, RandSpatialCropd, RandFlipd,
    NormalizeIntensityd, RandScaleIntensityd,
    RandShiftIntensityd, Orientationd, Spacingd, ToTensord,
)

MODALITY_KEYS = ['t1', 't1ce', 't2', 'flair']
ALL_KEYS = MODALITY_KEYS + ['label']


def get_train_transforms():
    """Full augmentation pipeline for training."""
    return Compose([
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
        Orientationd(keys=ALL_KEYS, axcodes='RAS'),
        Spacingd(keys=ALL_KEYS, pixdim=(1.0, 1.0, 1.0),
                 mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest')),
        CropForegroundd(keys=ALL_KEYS, source_key='t1'),
        RandSpatialCropd(keys=ALL_KEYS, roi_size=(128, 128, 64), random_size=False),
        RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys=MODALITY_KEYS, nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=MODALITY_KEYS, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=MODALITY_KEYS, offsets=0.1, prob=0.5),
        ToTensord(keys=ALL_KEYS),
    ])


def get_val_transforms():
    """Minimal transforms for validation (no augmentation)."""
    return Compose([
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
        Orientationd(keys=ALL_KEYS, axcodes='RAS'),
        Spacingd(keys=ALL_KEYS, pixdim=(1.0, 1.0, 1.0),
                 mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest')),
        NormalizeIntensityd(keys=MODALITY_KEYS, nonzero=True, channel_wise=True),
        ToTensord(keys=ALL_KEYS),
    ])


def get_data_dicts(data_dir, split='train', hospital_id=None, num_hospitals=3):
    """Scan BraTS directory and build file dictionaries per hospital partition."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'BraTS data not found at {data_dir}.')
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    split_idx = int(len(patient_dirs) * 0.8)
    patients = patient_dirs[:split_idx] if split == 'train' else patient_dirs[split_idx:]
    if hospital_id is not None:
        n = len(patients)
        size = n // num_hospitals
        start = (hospital_id - 1) * size
        end = start + size if hospital_id < num_hospitals else n
        patients = patients[start:end]
    return [{
        't1':    str(p / f'{p.name}_t1.nii.gz'),
        't1ce':  str(p / f'{p.name}_t1ce.nii.gz'),
        't2':    str(p / f'{p.name}_t2.nii.gz'),
        'flair': str(p / f'{p.name}_flair.nii.gz'),
        'label': str(p / f'{p.name}_seg.nii.gz'),
    } for p in patients]
