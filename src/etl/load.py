"""Load CSV input and fail early on malformed telemetry."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import (
    ID_COL,
    PRODUCT_ID_COL,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
    TYPE_COL,
)


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    """Load labelled training observations."""
    df = pd.read_csv(path if path is not None else TRAIN_CSV)
    validate_schema(df, require_target=True)
    return df


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    """Load observations for inference; failure flags are not required."""
    df = pd.read_csv(path if path is not None else TEST_CSV)
    validate_schema(df, require_target=False)
    return df


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Validate identifiers, finite numeric readings and equipment types."""
    required = [ID_COL, PRODUCT_ID_COL, TYPE_COL, *RAW_NUMERIC_FEATURES]
    if require_target:
        required.append(TARGET_COL)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Input dataset is empty")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names")
    for column in required:
        if df[column].isna().any():
            raise ValueError(f"Null values in column {column}")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate observation id")
    if not df[TYPE_COL].isin(["L", "M", "H"]).all():
        raise ValueError("Type must be one of L, M, H")
    if df[PRODUCT_ID_COL].astype(str).str.strip().eq("").any():
        raise ValueError("Product ID must not be blank")
    for column in RAW_NUMERIC_FEATURES:
        if not is_numeric_dtype(df[column]) or not np.isfinite(df[column]).all():
            raise ValueError(f"Expected finite numeric values in column {column}")
        if (df[column] < 0).any():
            raise ValueError(f"Negative telemetry in column {column}")
    if require_target and not df[TARGET_COL].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COL} must be binary (0 or 1)")
