# Run the locked environment from any working directory.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    poetry run ml-train --smoke
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
    poetry run ml-predict
    if ($LASTEXITCODE -ne 0) { throw "Inference failed" }
} finally {
    Pop-Location
}
