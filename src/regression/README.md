# Diamond price regression

This package contains only the regression workflow. It predicts `price`; `clarity`
is one of nine raw inputs.

- `cleaning.py` reproduces the historical 53,899-row regression population.
- `feature_engineering.py` contains the original base and R10 human features.
- `preprocessing.py` creates one shared 70/15/15 split and fits learned transforms on training rows.
- `model.py` defines the course price-regression ANN.
- `training.py` trains ANN and tree models, runs smoke validation, and selects by validation MAE.
- `evaluation.py` calculates dollar, percentage, severity, and price-band metrics.
- `visualizations.py` creates EDA, feature-importance, relationship, and error plots.
- `datasets.py` exports original-unit and exact processed datasets with statistics.
- `prediction.py` saves and loads the complete inference pipeline.
- `api.py` serves the saved pipeline locally on port 8766.

Run `notebooks/03_regression.ipynb` from top to bottom. Set the environment
variable `DIAMOND_SMOKE_TEST=1` only for a fast structural verification; smoke
models are deliberately small and must not be treated as final results.
