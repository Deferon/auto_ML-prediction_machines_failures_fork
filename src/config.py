"""Project defaults; paths can be overridden for deployed installations."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", PROJECT_ROOT / "keis7-main")).resolve()
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).resolve()
MLFLOW_DB = ARTIFACTS_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlartifacts"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}")
MLFLOW_ARTIFACTS_URI = MLFLOW_ARTIFACTS_DIR.as_uri()
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "machine_failure_prediction")

TARGET_COL = "Machine failure"
ID_COL = "id"
PRODUCT_ID_COL = "Product ID"
TYPE_COL = "Type"

FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "delta_temperature [K]",
    "Power [kW]",
    "air_mass",
    "air_heat_power [kW]",
    "efficiency [%]",
]

# Failure flags describe the outcome and must never be predictors.
MODEL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + [TYPE_COL]
CAT_FEATURES = [TYPE_COL]
FEATURE_SCHEMA_VERSION = 2

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "early_stopping_rounds": 100,
    "auto_class_weights": "Balanced",
}

TRAIN_TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOKE_SAMPLE_SIZE = 5000

RISK_THRESHOLDS = {
    "low": 0.45,
    "medium": 0.50,
}
