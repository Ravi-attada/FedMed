# Chaitanya Day 1 - chore(model): set up PyTorch MONAI environment and explore BraTS dataset structure

## Tasks Completed
* Downloaded and extracted the BraTS 2021 dataset (~13.4 GB).
* Explored the NIfTI files to understand the 4 MRI modalities (T1, T1ce, T2, FLAIR).
* Identified the 3 segmentation targets (NCR, ED, ET).
* Set up the Python virtual environment with PyTorch 2.1 and MONAI 1.3.0.

## Notes & Challenges
* The dataset is massive; we need to ensure the data loader is optimized to prevent RAM bottlenecks.
