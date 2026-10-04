import numpy as np
import pandas as pd
import pytest

from src.config import FAILURE_FLAGS, MODEL_FEATURES
from src.etl.features import build_features, get_feature_matrix


def test_power_and_efficiency(telemetry):
    result = build_features(telemetry)
    power = 40 * 1600 / 9550
    assert result["Power [kW]"].iloc[0] == pytest.approx(power)
    assert result["delta_temperature [K]"].iloc[0] == 9
    heat = 14.4 * power * 1.005 * 9 / 60
    assert result["efficiency [%]"].iloc[0] == pytest.approx(100 * power / (heat + power))
    assert "Power [kW]" not in telemetry


def test_zero_power_has_finite_features(telemetry):
    telemetry["Torque [Nm]"] = 0
    result, _ = get_feature_matrix(build_features(telemetry))
    assert np.isfinite(result.select_dtypes("number")).all().all()
    assert result["efficiency [%]"].iloc[0] == 0


def test_features_ignore_outcome_and_preserve_rows(telemetry):
    changed = telemetry.copy()
    changed["Machine failure"] = 1
    changed[FAILURE_FLAGS] = 1
    expected, _ = get_feature_matrix(build_features(telemetry))
    actual, _ = get_feature_matrix(build_features(changed))
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.index.tolist() == [42]
    assert not set(FAILURE_FLAGS) & set(MODEL_FEATURES)
    assert "total_failures_cum" not in MODEL_FEATURES


def test_features_independent_of_batch(telemetry):
    other = telemetry.copy()
    other["id"] = 8
    other["Tool wear [min]"] = 20
    batch = pd.concat([telemetry, other], ignore_index=True)
    alone, _ = get_feature_matrix(build_features(telemetry))
    together, _ = get_feature_matrix(build_features(batch))
    pd.testing.assert_frame_equal(alone.reset_index(drop=True), together.iloc[:1])
    assert build_features(batch)["id"].tolist() == [7, 8]


def test_inference_needs_no_target_or_failure_flags(telemetry):
    inference = telemetry.drop(columns=["Machine failure", *FAILURE_FLAGS])
    features, target = get_feature_matrix(build_features(inference, is_train=False), False)
    assert features.columns.tolist() == MODEL_FEATURES
    assert target is None
    with pytest.raises(ValueError, match="Missing target"):
        get_feature_matrix(build_features(inference))
