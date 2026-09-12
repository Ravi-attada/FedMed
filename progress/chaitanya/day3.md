# Chaitanya Day 3 - feat(model): complete 3D U-Net decoder path and full forward pass sanity check

## Tasks Completed
* Implemented the decoder path with transposed convolutions.
* Added skip connections to retain high-resolution spatial features from the encoder.
* Wrote a quick sanity check script to pass a dummy tensor `(1, 4, 128, 128, 64)` through the model.

## Notes & Challenges
* Model compiles and forward pass works. Parameter count is around 4.8M.
