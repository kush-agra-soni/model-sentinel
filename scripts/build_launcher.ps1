# Rebuilds launcher.exe in the repo root. Requires uv (https://docs.astral.sh/uv/).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uvx --python 3.11 pyinstaller==6.11.1 --onefile --console --name launcher `
        --distpath . --workpath build --specpath build launcher/launcher.py
    Remove-Item -Recurse -Force build
    Write-Host "Built $root\launcher.exe"
} finally {
    Pop-Location
}
