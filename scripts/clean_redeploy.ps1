# Full clean redeploy for Windows — stops services, wipes output/, reinstalls, restarts.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

Write-Host "==> Stopping services..."
Get-Process -Name "python","uvicorn" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*TradingAgents*" -or $_.CommandLine -match "api.main|webui.py" } |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "==> Wiping local runtime state (output/)..."
@("output\db", "output\analysis", "output\logs", "output\cache", "output\memory") | ForEach-Object {
    if (Test-Path $_) { Remove-Item -Recurse -Force $_ }
}
@("output\db", "output\analysis", "output\logs", "output\cache", "output\memory") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

Write-Host "==> Reinstalling package..."
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\pip.exe install -q -U pip
& .\.venv\Scripts\pip.exe install -q -e ".[webui,news,dev]"

if (-not (Test-Path ".env")) {
    Write-Error ".env missing — copy from .env.example and set DEEPSEEK_API_KEY"
}

Write-Host "==> Verifying DeepSeek key from .env..."
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
$key = $env:DEEPSEEK_API_KEY
if (-not $key) { Write-Error "DEEPSEEK_API_KEY is empty in .env" }
Write-Host ("DEEPSEEK key ends with: ...{0}" -f $key.Substring($key.Length - 4))
& .\.venv\Scripts\python.exe -c @"
import os
from openai import OpenAI
key = os.environ['DEEPSEEK_API_KEY']
client = OpenAI(api_key=key, base_url='https://api.deepseek.com')
r = client.chat.completions.create(model='deepseek-chat', messages=[{'role':'user','content':'Say OK'}], max_tokens=5)
print('DeepSeek auth OK:', r.choices[0].message.content.strip()[:20])
"@

Write-Host "==> Starting API on http://127.0.0.1:9000 ..."
Start-Process -FilePath ".\.venv\Scripts\uvicorn.exe" `
    -ArgumentList "api.main:app","--host","127.0.0.1","--port","9000" `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Normal

Write-Host "==> Starting Web UI on http://127.0.0.1:8501 ..."
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m","streamlit","run","webui.py","--server.address","127.0.0.1","--server.port","8501","--server.headless","true","--browser.gatherUsageStats","false" `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Normal

Start-Sleep -Seconds 10
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:9000/healthz" -UseBasicParsing | Out-Null
    Write-Host "API OK"
    $ui = Invoke-WebRequest -Uri "http://127.0.0.1:8501/" -UseBasicParsing
    Write-Host ("Web UI HTTP {0}" -f $ui.StatusCode)
} catch {
    Write-Warning "Health check failed — services may still be starting: $_"
}
Write-Host "==> Done. API: http://127.0.0.1:9000/ui  WebUI: http://127.0.0.1:8501"
