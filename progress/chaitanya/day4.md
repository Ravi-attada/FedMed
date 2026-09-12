# Chaitanya Day 4 - feat(train): implement centralized training loop with DiceCE loss and Adam optimizer

## Tasks Completed
* Built the `train_centralized.py` script to establish a baseline.
* Integrated `DiceCELoss` (combining Dice Loss and Cross-Entropy) from MONAI.
* Set up the Adam optimizer with `CosineAnnealingLR` scheduling.

## Notes & Challenges
* Balancing the Lambda weights for Dice vs CrossEntropy. Currently set to 0.5 each.
