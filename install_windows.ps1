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
    Write-Error "Python $requirePy required. Found $ver."
  }
}

function Ensure-PSQL {
  if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    $pgBin = Get-ChildItem -Path "C:\Program Files\PostgreSQL" -Filter psql.exe -Recurse -ErrorAction SilentlyContinue |
      Select-Object -First 1 -ExpandProperty DirectoryName
    if (-not $pgBin) {
      $pgBin = Get-ChildItem -Path "C:\Program Files (x86)\PostgreSQL" -Filter psql.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty DirectoryName
    }
    if ($pgBin) {
      $env:PATH = "$pgBin;$env:PATH"
      $script:PSQL_EXE = (Join-Path $pgBin "psql.exe")
    } else {
      Ensure-Winget
      Write-Host "psql not found. Installing PostgreSQL via winget..."
      winget source update | Out-Null
      winget install -e --id PostgreSQL.PostgreSQL --accept-package-agreements --accept-source-agreements

      $pgBin = Get-ChildItem -Path "C:\Program Files\PostgreSQL" -Filter psql.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty DirectoryName
      if ($pgBin) {
        $env:PATH = "$pgBin;$env:PATH"
        $script:PSQL_EXE = (Join-Path $pgBin "psql.exe")
      }
    }
  }

  if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    $manualBin = Read-Host "psql still not found. Enter PostgreSQL bin path (e.g., C:\Program Files\PostgreSQL\15\bin) or leave blank to abort"
    if (-not [string]::IsNullOrWhiteSpace($manualBin)) {
      if (Test-Path (Join-Path $manualBin "psql.exe")) {
        $env:PATH = "$manualBin;$env:PATH"
        $script:PSQL_EXE = (Join-Path $manualBin "psql.exe")
      } else {
        Write-Error "psql.exe not found in $manualBin"
      }
    } else {
      Write-Error "psql still not found. Install PostgreSQL and add its bin to PATH, then restart the terminal."
    }
  }

  if (-not $script:PSQL_EXE) {
    $cmd = Get-Command psql -ErrorAction SilentlyContinue
    if ($cmd) {
      $script:PSQL_EXE = $cmd.Source
    }
  }
}

function Update-EnvFile {
  param(
    [string]$Path,
    [string]$Key,
    [string]$Value
  )

  if (-not (Test-Path $Path)) {
    New-Item -Path $Path -ItemType File | Out-Null
  }

  $content = Get-Content -Path $Path -ErrorAction SilentlyContinue
  $pattern = "^${Key}="
  if ($content -match $pattern) {
    $content = $content -replace $pattern + ".*", ("{0}={1}" -f $Key, $Value)
    Set-Content -Path $Path -Value $content
  } else {
    Add-Content -Path $Path -Value ("{0}={1}" -f $Key, $Value)
  }
}

Set-Location $ProjectRoot

Ensure-Python
Ensure-PSQL

if (-not (Test-Path -Path "venv")) {
  Write-Host "Creating venv..."
  py -3.10 -m venv venv
} else {
  Write-Host "venv already exists. Skipping venv creation."
}

$activate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  Write-Error "Virtualenv activation script not found at $activate"
}

. $activate

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary

# ---- PostgreSQL setup ----
$DB_NAME = Read-Host "DB name"
if ([string]::IsNullOrWhiteSpace($DB_NAME)) { $DB_NAME = "cctv_logs" }

$DB_USER = Read-Host "DB user"
if ([string]::IsNullOrWhiteSpace($DB_USER)) { $DB_USER = "postgres" }

$DB_HOST = Read-Host "DB host"
if ([string]::IsNullOrWhiteSpace($DB_HOST)) { $DB_HOST = "localhost" }

$DB_PORT = Read-Host "DB port"
if ([string]::IsNullOrWhiteSpace($DB_PORT)) { $DB_PORT = "5432" }

$DB_PASS = Read-Host "DB password (will be saved to .env)" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($DB_PASS)
$DB_PASS_PLAIN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

$DB_URL = "postgresql://$DB_USER`:$DB_PASS_PLAIN@$DB_HOST`:$DB_PORT/$DB_NAME"

$envPath = Join-Path $ProjectRoot ".env"
Update-EnvFile -Path $envPath -Key "DATABASE_URL" -Value $DB_URL

$env:PGPASSWORD = $DB_PASS_PLAIN

$psqlExe = $script:PSQL_EXE
if (-not $psqlExe) { $psqlExe = "psql" }

$exists = & $psqlExe -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'"
if ($exists.Trim() -ne "1") {
  & $psqlExe -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE `"$DB_NAME`";"
}

$env:PGPASSWORD = $null

# Seed default admin user in users table
if (Test-Path "scripts\init_db.py") {
  Write-Host "Seeding default admin user..."
  python scripts\init_db.py
}

Write-Host "--------------------------------------------------------"
Write-Host "Install complete! PostgreSQL is required."
Write-Host "Database URL saved to .env as DATABASE_URL."
Write-Host "--------------------------------------------------------"
Write-Host "To run the system:"
Write-Host "1. Activate venv: .\venv\Scripts\Activate.ps1"
Write-Host "2. Start Web UI: python run.py"
Write-Host "--------------------------------------------------------"
