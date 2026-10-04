import numpy as np
import pandas as pd
import pytest

from src.monitoring import evaluate_drift, interpret_psi, population_stability_index


def test_interpret_psi_stable():
    assert interpret_psi(0.05) == "stable"
    assert interpret_psi(0.0) == "stable"


def test_interpret_psi_warning():
    assert interpret_psi(0.15) == "warning"


def test_interpret_psi_critical():
    assert interpret_psi(0.3) == "critical"


def test_evaluate_drift_all_stable():
    drift = {
        "Torque [Nm]": {"mean_shift": -0.01, "psi": 0.0002},
        "Tool wear [min]": {"mean_shift": -0.12, "psi": 0.00018},
    }
    result = evaluate_drift(drift)
    assert result["overall_status"] == "stable"
    assert result["alerts"] == []


def test_evaluate_drift_with_alert():
    drift = {
        "Torque [Nm]": {"mean_shift": 0.5, "psi": 0.3},
    }
    result = evaluate_drift(drift)
    assert result["overall_status"] == "critical"
    assert len(result["alerts"]) == 1


@pytest.mark.parametrize("values", [[1] * 10, [0, 1, 2, 3, 4]])
def test_identical_distributions_have_zero_psi(values):
    assert population_stability_index(pd.Series(values), pd.Series(values)) == pytest.approx(0)


@pytest.mark.parametrize("reference,current", [([0] * 10, [10] * 10), ([0, 1, 2], [30, 40, 50])])
def test_disjoint_distributions_produce_finite_alert(reference, current):
    psi = population_stability_index(pd.Series(reference), pd.Series(current))
    assert np.isfinite(psi)
    assert interpret_psi(psi) == "critical"


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_invalid_psi_input_rejected(values):
    with pytest.raises(ValueError):
        population_stability_index(pd.Series([1, 2]), pd.Series(values))
