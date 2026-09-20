# Flora Carbon AI - interactive local setup & run script (Windows PowerShell).
#
# Mirrors run.sh: handles a missing Python install, no venv yet, a stale venv,
# missing/partial dependencies, sample images not yet generated, optional
# Kaggle credentials, port already in use, and non-interactive sessions
# (piped input / CI) where prompts fall back to sane defaults.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Info($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "WARN  $msg" -ForegroundColor Yellow }
function Write-Fail($msg)  { Write-Host "FAIL  $msg" -ForegroundColor Red }

$IsInteractiveSession = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)

function Ask-YesNo($question, $defaultYes = $true) {
    if (-not $IsInteractiveSession) { return $defaultYes }
    $suffix = if ($defaultYes) { "[Y/n]" } else { "[y/N]" }
    $reply = Read-Host "$question $suffix"
    if ([string]::IsNullOrWhiteSpace($reply)) { return $defaultYes }
    return $reply -match '^[Yy]'
}

function Ask-Value($prompt, $default) {
    if (-not $IsInteractiveSession) { return $default }
    $reply = Read-Host "$prompt [$default]"
    if ([string]::IsNullOrWhiteSpace($reply)) { return $default }
    return $reply
}

Write-Host ""
Write-Host "Flora Carbon AI - Local Setup & Run" -ForegroundColor White
Write-Host "Working directory: $ScriptDir`n"

# ---- 1. Locate Python 3 -----------------------------------------------------

Write-Info "Checking for Python 3..."
$PythonBin = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $verOutput = & $candidate --version 2>&1
        if ($verOutput -match "Python 3") {
            $PythonBin = $candidate
            break
        }
    } catch { }
}

if (-not $PythonBin) {
    Write-Fail "No Python 3 interpreter found on PATH."
    Write-Host "Install Python 3.9+ from https://www.python.org/downloads/ and re-run this script."
    exit 1
}
Write-Ok "Found $(& $PythonBin --version) ($PythonBin)"

# ---- 2. Virtual environment --------------------------------------------------

$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

if (Test-Path $VenvPy) {
    Write-Ok "Existing virtual environment found at .venv"
    if (Ask-YesNo "Recreate it from scratch? (fixes a broken/stale venv)" $false) {
        Write-Info "Removing old .venv..."
        Remove-Item -Recurse -Force $VenvDir
    }
}

if (-not (Test-Path $VenvPy)) {
    Write-Info "Creating virtual environment at .venv ..."
    & $PythonBin -m venv $VenvDir
    if (-not (Test-Path $VenvPy)) {
        Write-Fail "Failed to create a venv."
        exit 1
    }
    Write-Ok "Virtual environment created."
}

# ---- 3. Dependencies ---------------------------------------------------------

Write-Info "Installing/upgrading dependencies from requirements.txt..."
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Dependency installation failed - see the pip output above."
    exit 1
}
Write-Ok "Dependencies installed."

# ---- 4. Sample images ---------------------------------------------------------

$SampleDir = Join-Path $ScriptDir "sample_images"
$SampleCount = 0
if (Test-Path $SampleDir) {
    $SampleCount = (Get-ChildItem $SampleDir -Include *.png,*.jpg -File -ErrorAction SilentlyContinue).Count
}

if ($SampleCount -lt 3) {
    Write-Info "Sample benchmark images missing/incomplete - generating them..."
    & $VenvPy create_samples.py
} else {
    Write-Ok "$SampleCount sample image(s) already present."
    if (Ask-YesNo "Regenerate sample images anyway?" $false) {
        & $VenvPy create_samples.py
    }
}

# ---- 5. Sync the decoupled static frontend build ------------------------------

if ((Test-Path "templates\index.html") -and (Test-Path "static-frontend\config.js")) {
    Write-Info "Syncing static-frontend/index.html with templates/index.html..."
    & $VenvPy -c @"
content = open('templates/index.html').read()
marker = '<script src="https://cdn.tailwindcss.com"></script>'
if marker in content and '<script src="config.js"></script>' not in content:
    content = content.replace(marker, '<script src="config.js"></script>' + chr(10) + '  ' + marker, 1)
open('static-frontend/index.html', 'w').write(content)
print('  static-frontend/index.html synced.')
"@
}

# ---- 6a. Hardware profile preview ---------------------------------------------

Write-Info "Detecting local hardware profile (drives adaptive resource limits)..."
& $VenvPy -c @"
import hardware
p = hardware.get_profile()
print(f"  RAM: {p['total_ram_gb']}GB | CPUs: {p['cpu_count']} | Tier: {p['tier']}")
for k, v in p['config'].items():
    print(f'    {k}: {v}')
"@

# ---- 6b. Optional Kaggle credentials -------------------------------------------

$KaggleJsonPath = Join-Path $HOME ".kaggle\kaggle.json"
$KaggleReady = $false
if ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY) {
    Write-Ok "Kaggle credentials found in environment variables - Kaggle Benchmark panel will be enabled."
    $KaggleReady = $true
} elseif (Test-Path $KaggleJsonPath) {
    Write-Ok "Kaggle credentials found at $KaggleJsonPath - Kaggle Benchmark panel will be enabled."
    $KaggleReady = $true
} else {
    Write-Warn "No Kaggle credentials found - the Kaggle Dataset Benchmark panel will show as unavailable."
    if (Ask-YesNo "Set up Kaggle credentials now? (from https://www.kaggle.com/settings -> API -> Create New Token)" $false) {
        $kaggleUser = Ask-Value "Kaggle username" ""
        $kaggleKey = Ask-Value "Kaggle API key" ""
        if ($kaggleUser -and $kaggleKey) {
            $kaggleDir = Join-Path $HOME ".kaggle"
            New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
            @{ username = $kaggleUser; key = $kaggleKey } | ConvertTo-Json | Set-Content -Path $KaggleJsonPath
            Write-Ok "Saved credentials to $KaggleJsonPath"
            $KaggleReady = $true
        } else {
            Write-Warn "Skipped - no username/key entered."
        }
    } else {
        Write-Host "  (You can enable this later - see the README 'Data & Credentials Required' section.)"
    }
}

# ---- 7. Optional: pre-fetch the Kaggle benchmark dataset -----------------------

if ($KaggleReady) {
    if (Ask-YesNo "Pre-fetch the Kaggle benchmark dataset now (mcagriaksoy/trees-in-satellite-imagery)?" $false) {
        Write-Info "Downloading via kagglehub (cached locally afterward)..."
        & $VenvPy -c @"
import kagglehub
path = kagglehub.dataset_download('mcagriaksoy/trees-in-satellite-imagery')
print(f'  Cached at: {path}')
"@
        if ($LASTEXITCODE -eq 0) { Write-Ok "Kaggle dataset ready." }
        else { Write-Warn "Kaggle dataset download failed - check your credentials/network. The app still works without it." }
    }
}

# ---- 8. Optional: build the Docker image locally --------------------------------

if (Get-Command docker -ErrorAction SilentlyContinue) {
    if (Ask-YesNo "Docker detected. Build the container image locally too? (optional, verifies the deployable build)" $false) {
        Write-Info "Building Docker image 'flora-carbon-ai:local'..."
        docker build -t flora-carbon-ai:local .
        if ($LASTEXITCODE -eq 0) { Write-Ok "Docker image built: flora-carbon-ai:local (run with: docker run -p 7860:7860 flora-carbon-ai:local)" }
        else { Write-Warn "Docker build failed - see the output above. This does not block running the app locally." }
    }
} else {
    Write-Warn "Docker not found on PATH - skipping the optional container build (not required to run locally)."
}

# ---- 9. Port selection & availability check --------------------------------------

$Port = Ask-Value "Port to run on" "7860"

function Test-PortInUse($p) {
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, [int]$p)
        $listener.Start()
        return $false
    } catch {
        return $true
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

while (Test-PortInUse $Port) {
    Write-Warn "Port $Port is already in use."
    if (Ask-YesNo "Try a different port?" $true) {
        $Port = Ask-Value "Port to run on" ([int]$Port + 1)
    } else {
        Write-Fail "Cannot start - port $Port is busy. Free it or choose another port."
        exit 1
    }
}
Write-Ok "Port $Port is available."

# ---- 10. Launch --------------------------------------------------------------------

$OpenBrowser = Ask-YesNo "Open the dashboard in your browser once the server is up?" $true

Write-Host ""
Write-Host "Starting Flora Carbon AI on http://127.0.0.1:$Port ..." -ForegroundColor White
Write-Host "Press Ctrl+C to stop.`n"

if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        param($p)
        Start-Sleep -Seconds 2
        Start-Process "http://127.0.0.1:$p"
    } -ArgumentList $Port | Out-Null
}

$env:PORT = $Port
& $VenvPy app.py
