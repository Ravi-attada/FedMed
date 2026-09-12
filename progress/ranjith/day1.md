# Ranjith Kumar Day 1 - chore(data): explore BraTS 2021 NIfTI structure and set up MONAI environment

## Tasks Completed
* Explored the NIfTI (`.nii.gz`) file formats from the BraTS dataset.
* Verified that `nibabel` can successfully load the 3D volumes and labels.
* Initialized the MONAI transforms sandbox.

## Notes & Challenges
* Header orientations in NIfTI files vary; we will need an Orientationd transform in the pipeline.
