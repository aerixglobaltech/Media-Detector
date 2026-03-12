param(
  [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$requirePy = "3.10.11"
$script:PSQL_EXE = $null

function Ensure-Winget {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Install App Installer from Microsoft Store."
  }
}

function Ensure-Python {
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Ensure-Winget
    Write-Host "Python launcher not found. Installing Python $requirePy via winget..."
    winget install -e --id Python.Python.3.10 --version $requirePy
  }

  $ver = & py -3.10 -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
  if ($ver.Trim() -ne $requirePy) {
    Write-Error "Python $requirePy required. Found $ver. Please install it."
  }
}

function Ensure-PSQL {
  if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    # Search common install paths
    $pgPaths = @(
        "C:\Program Files\PostgreSQL\*\bin",
        "C:\Program Files (x86)\PostgreSQL\*\bin"
    )
    
    foreach ($path in $pgPaths) {
        $found = Get-ChildItem -Path $path -Filter psql.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            $pgBin = $found.DirectoryName
            $env:PATH = "$pgBin;$env:PATH"
            $script:PSQL_EXE = $found.FullName
            break
        }
    }

    if (-not $script:PSQL_EXE) {
      Ensure-Winget
      Write-Host "psql not found. Installing PostgreSQL 16 via winget..."
      winget source update | Out-Null
      winget install -e --id PostgreSQL.PostgreSQL.16 --accept-package-agreements --accept-source-agreements
      
      # Re-search after install
      foreach ($path in $pgPaths) {
          $found = Get-ChildItem -Path $path -Filter psql.exe -ErrorAction SilentlyContinue | Select-Object -First 1
          if ($found) {
              $pgBin = $found.DirectoryName
              $env:PATH = "$pgBin;$env:PATH"
              $script:PSQL_EXE = $found.FullName
              break
          }
      }
    }
  }

  if (-not (Get-Command psql -ErrorAction SilentlyContinue) -and -not $script:PSQL_EXE) {
    Write-Host "WARNING: psql still not found in PATH."
    $manualBin = Read-Host "Enter PostgreSQL bin path manually (e.g., C:\Program Files\PostgreSQL\16\bin) or leave blank"
    if (-not [string]::IsNullOrWhiteSpace($manualBin)) {
      if (Test-Path (Join-Path $manualBin "psql.exe")) {
        $env:PATH = "$manualBin;$env:PATH"
        $script:PSQL_EXE = (Join-Path $manualBin "psql.exe")
      }
    }
  }

  if (-not $script:PSQL_EXE) {
     $cmd = Get-Command psql -ErrorAction SilentlyContinue
     if ($cmd) { $script:PSQL_EXE = $cmd.Source }
  }
}

function Update-EnvFile {
  param([string]$Path, [string]$Key, [string]$Value)
  if (-not (Test-Path $Path)) { New-Item -Path $Path -ItemType File | Out-Null }
  $content = Get-Content -Path $Path -ErrorAction SilentlyContinue
  $pattern = "^${Key}="
  if ($content -match $pattern) {
    $content = $content -replace $pattern + ".*", ("{0}={1}" -f $Key, $Value)
    Set-Content -Path $Path -Value $content
  } else {
    Add-Content -Path $Path -Value ("{0}={1}" -f $Key, $Value)
  }
}

# --- MAIN EXECUTION ---
Set-Location $ProjectRoot

Write-Host "--- Media Detector Installation ---" -ForegroundColor Cyan

Ensure-Python
Ensure-PSQL

if (-not (Test-Path -Path "venv")) {
  Write-Host "Creating Virtual Environment (venv)..."
  py -3.10 -m venv venv
}

$activate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
. $activate

Write-Host "Installing Dependencies (this may take a while)..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Explicitly ensure Postgres driver is installed
pip install psycopg2-binary

# ---- PostgreSQL Setup ----
Write-Host "`n--- Database Configuration ---" -ForegroundColor Yellow
$DB_NAME = Read-Host "Enter Database Name [default: mediadetect]"
if ([string]::IsNullOrWhiteSpace($DB_NAME)) { $DB_NAME = "mediadetect" }

$DB_USER = Read-Host "Enter PostgreSQL Username [default: postgres]"
if ([string]::IsNullOrWhiteSpace($DB_USER)) { $DB_USER = "postgres" }

$DB_HOST = Read-Host "Enter PostgreSQL Host [default: 127.0.0.1]"
if ([string]::IsNullOrWhiteSpace($DB_HOST)) { $DB_HOST = "127.0.0.1" }

$DB_PORT = Read-Host "Enter PostgreSQL Port [default: 5432]"
if ([string]::IsNullOrWhiteSpace($DB_PORT)) { $DB_PORT = "5432" }

$DB_PASS = Read-Host "Enter Password for $DB_USER" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($DB_PASS)
$DB_PASS_PLAIN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Build Connection URL
$DB_URL = "postgresql://$DB_USER`:$DB_PASS_PLAIN@$DB_HOST`:$DB_PORT/$DB_NAME"

# Update .env file
$envPath = Join-Path $ProjectRoot ".env"
Update-EnvFile -Path $envPath -Key "DATABASE_URL" -Value $DB_URL
Write-Host "Connection string saved to .env" -ForegroundColor Green

# Create Database if it doesn't exist
$env:PGPASSWORD = $DB_PASS_PLAIN
$psql = if ($script:PSQL_EXE) { $script:PSQL_EXE } else { "psql" }

Write-Host "Checking if database '$DB_NAME' exists..."
$exists = & $psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'"
if ($exists.Trim() -ne "1") {
    Write-Host "Creating database '$DB_NAME'..." -ForegroundColor Cyan
    & $psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE `"$DB_NAME`";"
}

# Initialize Schema
Write-Host "Initializing database tables..." -ForegroundColor Cyan
python -c "from app.db.session import init_db; init_db()"

# Migration support
if (Test-Path "migrate_to_pg.py") {
    $move = Read-Host "Do you want to migrate existing data from SQLite? (y/n) [default: n]"
    if ($move -eq "y") {
        python migrate_to_pg.py
    }
}

$env:PGPASSWORD = $null

# Seed Admin
if (Test-Path "scripts\seed_admin.py") {
  Write-Host "Seeding default admin user..."
  python scripts\seed_admin.py
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor Green
Write-Host "INSTALLATION SUCCESSFUL!" -ForegroundColor Green
Write-Host "1. Activate: .\venv\Scripts\Activate.ps1"
Write-Host "2. Run: python run.py"
Write-Host "--------------------------------------------------------"
