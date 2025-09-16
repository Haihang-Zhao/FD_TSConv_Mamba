# README

## Files

* **data\_loader.py**: Loads and concatenates `.npy` data/labels with optional normalization/standardization; provides a `Dataset` and splits into train/val/test `DataLoader`s.
* **models.py**: Uses a CNN encoder to extract frame features; bidirectional Mamba (forward/backward) models temporal dependencies; a linear head outputs classes.
* **train.py**: Runs training/validation/testing end-to-end and reports Accuracy/Precision/Recall/F1 and model size.
