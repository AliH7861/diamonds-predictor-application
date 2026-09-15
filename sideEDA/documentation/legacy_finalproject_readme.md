# Diamond Machine Learning Final Project

## Overview

This project contains two separate workflows built from the same raw diamond data:

- Classification predicts the clarity family.
- Regression predicts price and uses the original clarity grade as an input.

## Getting Started

1. For regression, activate `Diamond/.venv`, or install `requirements-regression.txt`.
2. Open the relevant notebook and use **Restart Kernel / Run All**.
3. Classification serves locally on port 8765; regression uses port 8766.

## Regression project structure

- `RegressionModel.ipynb`: the documented 18-section experiment and deployment workflow.
- `../src/regression/`: cleaning, features, preprocessing, training, evaluation, plots, exports, prediction, and API modules.
- `../data/regression/`: generated regression datasets and statistics after running the notebook.
- `../outputs/figures/regression/`: generated plots.
- `../outputs/reports/regression/`: validation comparisons and final test reports.
- `../models/regression/`: the selected model bundle and metadata.

The regression workflow reproduces the historical 53,899-row population and the
37,729 / 8,085 / 8,085 split. It trains CURRENT_BASELINE, HUMAN_ONLY, and
HUMAN_PLUS_RAW, then compares five algorithms on the validation-winning feature
set. The final winner is selected by validation MAE and evaluated once on test data.
Historical `$245.46` MAE is documentation, not a hardcoded result.

After a successful run, start the saved regression model from the Diamond folder:

```bash
python -m src.regression.api --port 8766
```

Send `POST /predict` nine raw inputs: carat, cut, color, clarity, depth, table,
x, y, and z. Do not send price because price is the target.

## Classification project structure

- `ClassificationModel.ipynb`: experiment setup, ANN/Random Forest/XGBoost training, validation comparison, and final test evaluation.
- `../src/classification/cleaning.py`: cleaning, validation, and clarity-family targets.
- `../src/classification/feature_engineering.py`: physical and price-relative features and experiment feature lists.
- `../src/classification/pipeline.py`: CSV loading, shared stratified split, and training-fitted transformations.
- `../src/classification/diagnostics.py`: exploration plots and top-eight feature relationships.
- `../src/classification/datasets.py`: DataFrame summaries and dataset exports.
- `../src/classification/prediction.py`: save/load the selected model and predict new diamonds.
- `../src/classification/api.py`: local HTTP prediction API.

Run the notebook from `Diamond` or any folder beneath it. Data defaults to
`Diamond/data/diamonds.csv`; you can pass another CSV path to
`prepare_classification_data`. Dependencies are pandas, NumPy, scikit-learn,
Matplotlib, TensorFlow, XGBoost, and a Jupyter/IPython notebook environment.

The refactor preserves feature formulas, model settings, and the 70/15/15 split.
It fixes the original use of `clean_df` before assignment and makes duplicate
removal cumulative (ignoring the CSV index column). Rerun all experiments for
current results. The original notebook and its saved outputs are backed up in
`.ipynb_checkpoints/ClassificationModel-before-refactor.ipynb`.

Optional diagnostics after loading `prepared`:

```python
from src.classification.diagnostics import show_diagnostics
show_diagnostics(prepared)
```

## Purpose

This project is for educational purposes.

## Classification scope and top-eight feature graphs

All modules in `../src/classification/` belong to diamond clarity classification.
They are not the regression pipeline. `RegressionModel.ipynb` remains separate.
The classification target is `Clarity_Target`, representing the ordered families
I, SI, VS, VVS, and IF.

After each experiment, the notebook selects that experiment's highest validation
Macro F1 model and calls `plot_top_features` from `src.classification.diagnostics`.
Features are ranked by the drop in validation Macro F1 when their values are
shuffled (five repeats on a reproducible, stratified sample of up to 2,000 rows).
Categorical features are shuffled before encoding so all their encoded columns
are considered together. The graphs show the top eight rankings and their
relationship with observed clarity: numeric boxplots in original feature units,
and categorical proportions within each clarity family.

These are model-specific predictive associations. Correlated features can share
importance, and negative drops indicate no measured benefit in this check.
Engineered features are shuffled separately from their source features. No test
rows are used for these plots. Run all notebook cells to generate the graphs
using the trained experiment models; full model training has not been rerun
as part of this refactor.

## End-to-end: datasets, training, and prediction API

Open `ClassificationModel.ipynb`, choose your ML Python environment, and use
**Restart Kernel / Run All**. The notebook will:

1. Prepare the data and export 14 datasets plus statistics CSVs to `../data/classification/`.
2. Display all DataFrames, data types, missing values, descriptive statistics,
   training distributions, and correlation matrices.
3. Train all nine experiment/model combinations and show top-eight feature graphs.
4. Select the model with the highest validation Macro F1 and evaluate it on test data.
5. Save that model, its fitted preprocessor, and metadata to `../models/classification/`.
6. Demonstrate a Python prediction and start a local HTTP API on port 8765.
7. Send an example HTTP prediction request and display the result.

Model artifacts are created only after successful training. No best model is
claimed or created from untrained experiments. The API starts after those artifacts
exist and serves the selected model, without retraining it. Keep the notebook
kernel alive, or start the API independently from the Diamond directory:

```bash
python -m src.classification.api --port 8765
```

Use the same environment as training (including joblib). The API binds to localhost.
`GET /model` reports the selected model and required inputs. `GET /health` checks
readiness. `POST /predict` accepts a raw diamond object or a list of 1–1000 objects:

```json
{"carat":0.7,"cut":"Ideal","color":"G","depth":61.5,"table":57,"x":5.7,"y":5.72,"z":3.51,"price":2500}
```

Required inputs are carat, depth, table, x, y, z, cut, and color. Price is also
required for experiments 2 and 3. All numeric inputs must be positive and finite.
Cut must be Fair, Good, Very Good, Premium, or Ideal; color must be D through J.
Do not send clarity: that is what the model predicts. Responses contain the
predicted clarity family and class probabilities (not calibrated confidence).
New rows receive feature engineering and the saved preprocessing; they are not
subjected to batch deduplication or dataset-wide outlier filtering.

For Python access after training:

```python
from src.classification.prediction import load_best_model, predict_diamonds
bundle = load_best_model()
predictions = predict_diamonds(bundle, {
    "carat": 0.7, "cut": "Ideal", "color": "G", "depth": 61.5,
    "table": 57, "x": 5.7, "y": 5.72, "z": 3.51, "price": 2500
})
```

## Dataset folders

```text
data/classification/
  cleaned/
    dataset.csv
    statistics/summary.csv
  engineered/
    dataset.csv
    statistics/summary.csv
  experiments/
    experiment_1/
      dataset.csv
      processed/{train,valid,test}.csv
      statistics/{dataset,train,valid,test}.csv
    experiment_2/  (same layout)
    experiment_3/  (same layout)
  manifest.json
  README.md
```

The notebook exporter preserves this layout on every run. Dataset contents and
split assignments are unchanged; only their locations have changed.

## Earlier EDA work

See [sideEDA](../sideEDA/README.md) for earlier CSV versions, cleaning diagnostics,
Experiment M reports, notebook snapshots, and a runnable EDA overview. The current
classifier still reads `../data/diamonds.csv`; historical cleaned versions are
separate under `../sideEDA/data/cleaned/`.
