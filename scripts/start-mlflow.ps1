$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $trackingUri = poetry run python -c "from src.config import MLFLOW_TRACKING_URI; print(MLFLOW_TRACKING_URI)"
    if ($LASTEXITCODE -ne 0) { throw "Run poetry sync --with dev first" }
    $artifactRoot = poetry run python -c "from src.config import MLFLOW_ARTIFACTS_DIR; print(MLFLOW_ARTIFACTS_DIR)"
    if ($LASTEXITCODE -ne 0) { throw "Cannot read artifact configuration" }
    poetry run mlflow server --backend-store-uri $trackingUri --default-artifact-root $artifactRoot --host 127.0.0.1 --port 5000
    if ($LASTEXITCODE -ne 0) { throw "MLflow server failed" }
} finally {
    Pop-Location
}
