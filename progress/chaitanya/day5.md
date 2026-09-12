# Chaitanya Day 5 - feat(train): add mixed precision FP16 training and run 10-epoch baseline experiment

## Tasks Completed
* Upgraded training loop to use PyTorch AMP (Automatic Mixed Precision).
* Added `torch.cuda.amp.autocast` to drastically reduce GPU memory usage.
* Ran a 10-epoch baseline training test to verify loss convergence.

## Notes & Challenges
* AMP reduced VRAM usage by almost 40%, allowing us to increase the SlidingWindow inferer batch size.
