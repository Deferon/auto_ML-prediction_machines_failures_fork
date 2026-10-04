"""Row-local telemetry transformations shared by training and inference."""

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, TARGET_COL, TYPE_COL


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Preserve row order and cardinality without using labels or failure flags.

    The is_train argument is retained for compatibility; both paths are identical.
    """
    out = df.copy()
    out["delta_temperature [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [kW]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] / 9550
    out["air_mass"] = 14.4 * out["Power [kW]"]
    out["air_heat_power [kW]"] = out["air_mass"] * 1.005 * out["delta_temperature [K]"] / 60
    denominator = out["air_heat_power [kW]"] + out["Power [kW]"]
    out["efficiency [%]"] = (
        (100 * out["Power [kW]"] / denominator.replace(0, np.nan)).fillna(0.0).clip(0.0, 100.0)
    )
    return out


def get_feature_matrix(
    df: pd.DataFrame, include_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Select the ordered model schema and optional target."""
    missing = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Features not built: {missing}")
    if include_target and TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    features = df[MODEL_FEATURES].copy()
    features[TYPE_COL] = features[TYPE_COL].astype(str)
    target = df[TARGET_COL].astype(int) if include_target else None
    return features, target
