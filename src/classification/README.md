# Classification workflow

This package predicts five clarity families: `I`, `SI`, `VS`, `VVS`, and `IF`.

- `cleaning.py`: schema checks, cleaning, and target mapping.
- `feature_engineering.py`: experiment features and feature lists.
- `preprocessing.py`: deterministic split, imputation, scaling, and encoding.
- `model.py`: course ANN architecture and the preserved Y17 tuned ANN definition.
- `training.py`: ANN and benchmark training, smoke validation, selection, and test evaluation.
- `evaluation.py`: classification metrics.
- `prediction.py` and `api.py`: saved-model inference.

The maintained experiment notebook is `notebooks/02_classification.ipynb`.
