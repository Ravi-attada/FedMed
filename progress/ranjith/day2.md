# Ranjith Kumar Day 2 - feat(data): implement 10-step MONAI transform pipeline with intensity normalization

## Tasks Completed
* Wrote the `preprocess.py` MONAI transform pipeline.
* Added `LoadImaged`, `EnsureChannelFirstd`, and `NormalizeIntensityd`.
* Implemented `RandSpatialCropd` and `RandFlipd` for training data augmentation.

## Notes & Challenges
* Intensity normalization is crucial for MRI. Using Z-score normalization for T1/T2 modalities.
