import numpy as np
import pandas as pd
import pytest

from src.config import FAILURE_FLAGS
from src.etl.load import load_train, validate_schema


def test_load_train_schema():
    frame = load_train()
    assert len(frame) > 1000
    assert frame["Machine failure"].isin([0, 1]).all()


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("Torque [Nm]", np.nan, "Null values"),
        ("Torque [Nm]", np.inf, "finite numeric"),
        ("Torque [Nm]", "bad", "finite numeric"),
        ("Tool wear [min]", -1, "Negative"),
        ("Type", "unknown", "Type must"),
        ("Machine failure", 2, "binary"),
        ("Product ID", " ", "blank"),
    ],
)
def test_invalid_values(telemetry, column, value, message):
    telemetry[column] = value
    with pytest.raises(ValueError, match=message):
        validate_schema(telemetry)


def test_missing_column(telemetry):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(telemetry.drop(columns=["Torque [Nm]"]))


def test_duplicate_observation_id(telemetry):
    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_schema(pd.concat([telemetry, telemetry], ignore_index=True))


def test_empty_dataset(telemetry):
    with pytest.raises(ValueError, match="empty"):
        validate_schema(telemetry.iloc[:0])


def test_flags_not_required(telemetry):
    validate_schema(telemetry.drop(columns=FAILURE_FLAGS))
