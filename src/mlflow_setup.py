"""Configure MLflow without deleting or rewriting existing tracking stores."""

from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.exceptions import MlflowException

from src.config import ARTIFACTS_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def setup_mlflow(
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Reuse experiments; let remote servers choose their artifact location."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    artifact_uri = None
    if urlparse(uri).scheme not in {"http", "https", "databricks"}:
        local_artifacts = (artifact_dir or ARTIFACTS_DIR / "mlartifacts").resolve()
        local_artifacts.mkdir(parents=True, exist_ok=True)
        artifact_uri = local_artifacts.as_uri()
    if mlflow.get_experiment_by_name(experiment_name) is None:
        try:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        except MlflowException:
            # Another worker may have created it between lookup and creation.
            if mlflow.get_experiment_by_name(experiment_name) is None:
                raise
    mlflow.set_experiment(experiment_name)
