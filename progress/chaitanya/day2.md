# Chaitanya Day 2 - feat(model): implement 3D U-Net encoder downsampling path with skip connections

## Tasks Completed
* Designed the 3D U-Net encoder architecture.
* Configured convolutional layers with Instance Normalization and LeakyReLU activations.
* Verified tensor shapes during downsampling.

## Notes & Challenges
* Decided to use InstanceNorm instead of BatchNorm since our batch sizes will be small (often 1 or 2) due to 3D memory constraints.
