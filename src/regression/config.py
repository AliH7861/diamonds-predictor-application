"""
Regression configuration, paths, seed, and tuned tree parameters.

Models predict log1p(price); use expm1() for dollar predictions.
"""
from config import PROJECT_ROOT as PROJECT_ROOT
from config import RANDOM_STATE as RANDOM_STATE
from config import RAW_DATA_PATH as RAW_DATA_PATH

# Use the shared project dataset path for regression.
DATA_PATH = RAW_DATA_PATH

# Recovered XGBoost parameters from the tuned R10 experiment.
XGB_PARAMS = {
    "objective": "reg:squarederror", "n_estimators": 5000,
    "learning_rate": 0.01898431564785491, "max_depth": 10,
    "min_child_weight": 11, "subsample": 0.8108402455121476,
    "colsample_bytree": 0.8314087516230554, "reg_alpha": 0.0017200908570583246,
    "reg_lambda": 0.40718839386498884, "gamma": 0.003072339007521299,
    "max_bin": 512, "tree_method": "hist", "eval_metric": "rmse",
    "early_stopping_rounds": 120, "random_state": RANDOM_STATE, "n_jobs": -1}

# Recovered Random Forest parameters from the tuned R10 experiment.
RF_PARAMS = {
    "n_estimators": 400, "max_depth": 28,
    "min_samples_split": 2, "min_samples_leaf": 1,
    "max_features": 0.8, "bootstrap": True,
    "random_state": RANDOM_STATE, "n_jobs": -1}
