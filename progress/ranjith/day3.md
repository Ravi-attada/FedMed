# Ranjith Kumar Day 3 - feat(data): implement BraTSDataset with CacheDataset and hospital partition support

## Tasks Completed
* Implemented `BraTSDataset` using MONAI CacheDataset to speed up data loading.
* Wrote a hospital partitioning script to simulate data distribution across 3 different hospital clients.

## Notes & Challenges
* Caching the first few transforms saves massive I/O overhead during training.
