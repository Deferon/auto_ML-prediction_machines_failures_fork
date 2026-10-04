import json

import pandas as pd
import pytest

from src.config import FAILURE_FLAGS, MODEL_FEATURES, SMOKE_SAMPLE_SIZE
from src.etl.load import load_test
from src.predict import predict
from src.train import train_model


@pytest.mark.parametrize("sample_size", [0, -1, 1])
def test_invalid_sample_size(sample_size, tmp_path):
    with pytest.raises(ValueError, match="at least 10"):
        train_model(sample_size=sample_size, output_dir=tmp_path)


def test_one_class_fails_before_tracking(telemetry, tmp_path):
    path = tmp_path / "train.csv"
    telemetry.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both target classes"):
        train_model(data_path=path, output_dir=tmp_path)
    assert not (tmp_path / "mlflow.db").exists()


@pytest.mark.slow
def test_train_reload_and_predict(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    metrics = train_model(sample_size=SMOKE_SAMPLE_SIZE, output_dir=tmp_path, tracking_uri=uri)
    assert 0.5 < metrics["roc_auc"] <= 1
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["f1"] == metrics["f1"]
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert metadata["features"] == MODEL_FEATURES
    assert metadata["training_rows"] + metadata["validation_rows"] == SMOKE_SAMPLE_SIZE

    batch = load_test().head(20).drop(columns=FAILURE_FLAGS)
    data_path = tmp_path / "test.csv"
    batch.to_csv(data_path, index=False)
    result = predict(
        model_path=tmp_path / "model.cbm",
        output_dir=tmp_path / "new-output",
        data_path=data_path,
        use_train_for_drift=False,
    )
    assert result["id"].tolist() == batch["id"].tolist()
    assert result["failure_probability"].between(0, 1).all()
    assert len(pd.read_csv(tmp_path / "new-output" / "predictions.csv")) == len(batch)
    report = json.loads((tmp_path / "new-output" / "inference_monitoring.json").read_text())
    assert report["drift"]["overall_status"] == "not_evaluated"
    assert report["performance"]["rows_per_sec"] > 0

    batch.head(1).to_csv(data_path, index=False)
    single = predict(
        model_path=tmp_path / "model.cbm",
        output_dir=tmp_path / "single",
        data_path=data_path,
        use_train_for_drift=True,
    )
    assert single["failure_probability"].iloc[0] == pytest.approx(
        result["failure_probability"].iloc[0]
    )
    assert single["risk_level"].iloc[0] == result["risk_level"].iloc[0]
    report = json.loads((tmp_path / "single" / "inference_monitoring.json").read_text())
    assert report["drift"]["features"]
