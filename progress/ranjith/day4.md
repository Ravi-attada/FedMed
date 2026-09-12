# Ranjith Kumar Day 4 - feat(eval): implement Dice score and HD95 metrics with BraTS sub-region evaluation

## Tasks Completed
* Implemented the evaluation metrics (`DiceMetric` and `HausdorffDistanceMetric`).
* Mapped the raw labels (1, 2, 4) to the standard BraTS sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

## Notes & Challenges
* HD95 computation is slow on CPU; ensuring it only runs during validation, not every training step.
