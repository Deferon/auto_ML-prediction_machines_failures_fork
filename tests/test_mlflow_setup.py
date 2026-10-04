from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from src import mlflow_setup


def test_existing_experiment_preserved(monkeypatch, tmp_path):
    store = tmp_path / "mlflow.db"
    store.write_bytes(b"existing tracking history")
    api = Mock()
    api.get_experiment_by_name.return_value = Mock(artifact_location="file:///old/location")
    monkeypatch.setattr(mlflow_setup, "mlflow", api)
    mlflow_setup.setup_mlflow(f"sqlite:///{store.as_posix()}", tmp_path / "artifacts")
    assert store.read_bytes() == b"existing tracking history"
    api.create_experiment.assert_not_called()


def test_remote_server_owns_artifacts(monkeypatch, tmp_path):
    api = Mock()
    api.get_experiment_by_name.return_value = None
    monkeypatch.setattr(mlflow_setup, "mlflow", api)
    mlflow_setup.setup_mlflow("https://tracking.example", tmp_path / "unused")
    api.create_experiment.assert_called_once_with(
        "machine_failure_prediction", artifact_location=None
    )
    assert not (tmp_path / "unused").exists()


def test_creation_error_is_not_swallowed(monkeypatch):
    api = Mock()
    api.get_experiment_by_name.return_value = None
    api.create_experiment.side_effect = MlflowException("permission denied")
    monkeypatch.setattr(mlflow_setup, "mlflow", api)
    with pytest.raises(MlflowException, match="permission denied"):
        mlflow_setup.setup_mlflow("https://tracking.example")
