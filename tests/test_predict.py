import pandas as pd
import pytest

from src.predict import assign_risk_level_vector, predict


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "Низкий"),
        (0.449, "Низкий"),
        (0.45, "Средний"),
        (0.499, "Средний"),
        (0.50, "Высокий"),
        (1, "Высокий"),
    ],
)
def test_fixed_risk_thresholds(score, expected):
    assert assign_risk_level_vector(pd.Series([score], index=[42])).to_dict() == {42: expected}


def test_tied_probabilities():
    assert assign_risk_level_vector(pd.Series([0.6] * 5)).tolist() == ["Высокий"] * 5


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities(score):
    with pytest.raises(ValueError, match="Probabilities"):
        assign_risk_level_vector(pd.Series([score]))


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict(model_path=tmp_path / "absent.cbm", use_train_for_drift=False)
