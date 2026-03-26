# BUILD_PORTABLE_PACKAGE.ps1
# -----------------------------------------------------------------------------
# RUN THIS SCRIPT TO RECREATE THE PORTABLE DEPENDENCIES FOR YOUR SETUP.EXE
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# 1. Configuration
$PKG_ROOT = Join-Path $PSScriptRoot "package"
$PY_DIR = Join-Path $PKG_ROOT "python"
$PG_DIR = Join-Path $PKG_ROOT "postgres"

# URLS
$PY_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
$PG_URL = "https://get.enterprisedb.com/postgresql/postgresql-16.2-1-windows-x64-binaries.zip"
$PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# 2. Create Folder Structure
Write-Host ">>> Creating package folder..." -ForegroundColor Cyan
if (!(Test-Path $PKG_ROOT)) { New-Item -ItemType Directory -Path $PKG_ROOT }
if (!(Test-Path $PY_DIR)) { New-Item -ItemType Directory -Path $PY_DIR }
if (!(Test-Path $PG_DIR)) { New-Item -ItemType Directory -Path $PG_DIR }

# 3. Download & Extract Python
if (!(Test-Path "$PY_DIR\python.exe")) {
    Write-Host ">>> Downloading Portable Python..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PY_URL -OutFile "$PKG_ROOT\python.zip"
    Expand-Archive -Path "$PKG_ROOT\python.zip" -DestinationPath $PY_DIR -Force
    Remove-Item "$PKG_ROOT\python.zip"

    # Headless support: Copy python.exe to pythonw.exe
    Copy-Item "$PY_DIR\python.exe" "$PY_DIR\pythonw.exe" -Force

    # Fix Python path for isolated environment
    $PTH_FILE = "$PY_DIR\python310._pth"
    # Overwrite the default .pth to include project root and enable site-packages
    $PTH_CONTENT = "python310.zip`r`n.`r`nimport site`r`n..\..`r`n"
    Set-Content -Path $PTH_FILE -Value $PTH_CONTENT
}

# 4. Download & Extract Postgres
if (!(Test-Path "$PG_DIR\bin\pg_ctl.exe")) {
    Write-Host ">>> Downloading Portable PostgreSQL..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PG_URL -OutFile "$PKG_ROOT\postgres.zip"
    Expand-Archive -Path "$PKG_ROOT\postgres.zip" -DestinationPath "$PKG_ROOT\temp_pg"
    Copy-Item -Path "$PKG_ROOT\temp_pg\pgsql\*" -Destination $PG_DIR -Recurse -Force
    Remove-Item "$PKG_ROOT\postgres.zip"
    Remove-Item -Path "$PKG_ROOT\temp_pg" -Recurse -Force
}

# 5. Install PIP & Project Requirements
Write-Host ">>> Installing Python requirements into portable folder..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $PIP_URL -OutFile "$PY_DIR\get-pip.py"
& "$PY_DIR\python.exe" "$PY_DIR\get-pip.py" --no-warn-script-location
Remove-Item "$PY_DIR\get-pip.py"

# Install requirements
& "$PY_DIR\python.exe" -m pip install -r "$PSScriptRoot\requirements.txt" --no-warn-script-location

Write-Host "`n[SUCCESS] Portable Bundle Created in \package\" -ForegroundColor Green
Write-Host "You can now run 'Inno Setup' on installer.iss!" -ForegroundColor Cyan
