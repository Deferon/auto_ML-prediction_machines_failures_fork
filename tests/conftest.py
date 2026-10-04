"""Small labelled fixtures; no network or persistent training artifacts."""

import pandas as pd
import pytest


@pytest.fixture
def telemetry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 7,
                "Product ID": "L50096",
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1600,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        ],
        index=[42],
    )
