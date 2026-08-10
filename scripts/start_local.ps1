# Start TradingAgents API + Web UI on Windows with .env loaded.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

if (-not (Test-Path ".env")) {
    Write-Error ".env missing"
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

New-Item -ItemType Directory -Force -Path "output\db","output\analysis","output\logs","output\cache","output\memory" | Out-Null

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error ".venv missing. Run: python -m venv .venv; then .\.venv\Scripts\python.exe -m pip install -e .[webui,news,dev]"
}

Write-Host "Starting API on http://127.0.0.1:9000 ..."
Start-Process -FilePath $py `
    -ArgumentList "-m","uvicorn","api.main:app","--host","127.0.0.1","--port","9000" `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Normal

Write-Host "Starting Web UI on http://127.0.0.1:8501 ..."
Start-Process -FilePath $py `
    -ArgumentList "-m","streamlit","run","webui.py","--server.address","127.0.0.1","--server.port","8501","--server.headless","true","--browser.gatherUsageStats","false" `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Normal

Start-Sleep -Seconds 12
try {
    $api = Invoke-WebRequest -Uri "http://127.0.0.1:9000/healthz" -UseBasicParsing -TimeoutSec 10
    Write-Host ("API OK: {0}" -f $api.Content)
    $ui = Invoke-WebRequest -Uri "http://127.0.0.1:8501/" -UseBasicParsing -TimeoutSec 10
    Write-Host ("Web UI HTTP {0}" -f $ui.StatusCode)
} catch {
    Write-Warning "Health check failed - check the two new terminal windows for errors: $_"
}
Write-Host "Done. API: http://127.0.0.1:9000/ui  WebUI: http://127.0.0.1:8501"
