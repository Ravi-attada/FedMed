"""
Member 2 - Ranjith (ML Engineer)
Task: BraTS Dataset class and DataLoaders
"""
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from data.preprocess import get_data_dicts, get_train_transforms, get_val_transforms


class BraTSDataset:
    """BraTS 2021 Dataset using MONAI CacheDataset for faster loading."""
    def __init__(self, data_dir, split='train', hospital_id=None, cache_rate=0.5, num_workers=4):
        data_dicts = get_data_dicts(data_dir, split=split, hospital_id=hospital_id)
        transforms = get_train_transforms() if split == 'train' else get_val_transforms()
        self.dataset = CacheDataset(data=data_dicts, transform=transforms,
                                    cache_rate=cache_rate, num_workers=num_workers)
        print(f'[Dataset] Hospital {hospital_id or "ALL"} | {split} | {len(data_dicts)} samples')

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]


def get_dataloaders(data_dir, batch_size=1, num_workers=4, hospital_id=None, cache_rate=0.5):
    """Returns (train_loader, val_loader) for a hospital node."""
    train_ds = BraTSDataset(data_dir, 'train', hospital_id, cache_rate, num_workers)
    val_ds = BraTSDataset(data_dir, 'val', hospital_id, 0.0, num_workers)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
