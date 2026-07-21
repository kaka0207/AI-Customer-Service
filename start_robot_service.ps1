$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$streamlitPath = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
$appPath = Join-Path $projectRoot "app.py"

function Pause-OnExit {
    param(
        [string]$Message = "Press Enter to close"
    )

    Write-Host ""
    Read-Host $Message | Out-Null
}

try {
    Set-Location $projectRoot

    if (-not (Test-Path $streamlitPath)) {
        throw "Missing Streamlit executable: $streamlitPath"
    }

    if (-not (Test-Path $appPath)) {
        throw "Missing app entry file: $appPath"
    }

    if (-not $env:DASHSCOPE_API_KEY) {
        $env:DASHSCOPE_API_KEY = Read-Host "Enter DASHSCOPE_API_KEY"
    }

    if (-not $env:DASHSCOPE_API_KEY) {
        throw "DASHSCOPE_API_KEY is required before startup."
    }

    Write-Host "Project root: $projectRoot"
    Write-Host "Launching Streamlit..."
    Write-Host "Browser URL: http://localhost:8501"
    Write-Host ""

    & $streamlitPath run $appPath
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "Streamlit exited with code $exitCode."
    }
}
catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    Pause-OnExit
    exit 1
}

Pause-OnExit
