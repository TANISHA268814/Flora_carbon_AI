#!/usr/bin/env bash
#
# Flora Carbon AI - interactive local setup & run script (macOS/Linux).
#
# Handles every "first run on this machine" scenario: missing Python, no
# venv yet, an existing-but-stale venv, missing/partial dependencies, sample
# images not yet generated, optional Kaggle credentials, port already in
# use, and non-interactive environments (CI/piped input) where prompts are
# skipped in favor of sane defaults.
#
set -uo pipefail

# ---- helpers -----------------------------------------------------------

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
info()  { printf "${BOLD}==>${RESET} %s\n" "$1"; }
ok()    { printf "${GREEN}  OK${RESET}  %s\n" "$1"; }
warn()  { printf "${YELLOW}WARN${RESET}  %s\n" "$1"; }
fail()  { printf "${RED}FAIL${RESET}  %s\n" "$1"; }

# True only when stdin is an interactive terminal - lets the script run
# unattended (CI, piped input, `yes | ./run.sh`) without hanging on a prompt.
is_interactive() { [ -t 0 ]; }

ask_yes_no() {
    # ask_yes_no "question" "default(y/n)"
    local question="$1" default="${2:-y}" reply
    if ! is_interactive; then
        [ "$default" = "y" ] && return 0 || return 1
    fi
    local prompt="[y/n]"
    [ "$default" = "y" ] && prompt="[Y/n]" || prompt="[y/N]"
    read -r -p "$question $prompt " reply
    reply="${reply:-$default}"
    case "$reply" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

ask_value() {
    # ask_value "prompt" "default"
    local prompt="$1" default="${2:-}" reply
    if ! is_interactive; then
        printf "%s" "$default"
        return
    fi
    read -r -p "$prompt [$default]: " reply
    printf "%s" "${reply:-$default}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { fail "Could not cd into $SCRIPT_DIR"; exit 1; }

printf "\n${BOLD}Flora Carbon AI - Local Setup & Run${RESET}\n"
printf "Working directory: %s\n\n" "$SCRIPT_DIR"

# ---- 1. Locate a usable Python 3 ----------------------------------------

info "Checking for Python 3..."
PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver="$("$candidate" -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
        if [ "$ver" = "3" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    fail "No Python 3 interpreter found on PATH."
    echo "Install Python 3.9+ from https://www.python.org/downloads/ and re-run this script."
    exit 1
fi
PY_VERSION="$("$PYTHON_BIN" --version 2>&1)"
ok "Found $PY_VERSION ($PYTHON_BIN)"

# ---- 2. Virtual environment ----------------------------------------------

VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    ok "Existing virtual environment found at .venv"
    if ask_yes_no "Recreate it from scratch? (fixes a broken/stale venv)" n; then
        info "Removing old .venv..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at .venv ..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        fail "Failed to create a venv. Is the 'venv' module available? (Debian/Ubuntu: apt install python3-venv)"
        exit 1
    fi
    ok "Virtual environment created."
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ---- 3. Dependencies -------------------------------------------------------

info "Checking installed dependencies against requirements.txt..."
MISSING=0
while IFS= read -r line; do
    # Strip version specifiers/comments/blank lines for a quick import-style check.
    pkg="$(echo "$line" | sed -E 's/[<>=!~].*//' | sed 's/#.*//' | xargs)"
    [ -z "$pkg" ] && continue
    mod="$pkg"
    case "$pkg" in
        pillow) mod="PIL" ;;
        opencv-python-headless) mod="cv2" ;;
        python-multipart) mod="multipart" ;;
    esac
    if ! "$VENV_PY" -c "import $mod" >/dev/null 2>&1; then
        MISSING=1
        break
    fi
done < requirements.txt

if [ "$MISSING" = "1" ] || ask_yes_no "Re-check complete. Install/upgrade all dependencies now?" y; then
    info "Installing dependencies (this can take a minute on first run)..."
    "$VENV_PIP" install --quiet --upgrade pip
    if ! "$VENV_PIP" install --quiet -r requirements.txt; then
        fail "Dependency installation failed - see the pip output above."
        exit 1
    fi
    ok "Dependencies installed."
else
    ok "All required packages already importable - skipping install."
fi

# ---- 4. Sample images -------------------------------------------------------

SAMPLE_DIR="$SCRIPT_DIR/sample_images"
SAMPLE_COUNT=0
if [ -d "$SAMPLE_DIR" ]; then
    SAMPLE_COUNT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' \) | wc -l | tr -d ' ')
fi

if [ "$SAMPLE_COUNT" -lt 3 ]; then
    info "Sample benchmark images missing/incomplete - generating them..."
    "$VENV_PY" create_samples.py
else
    ok "$SAMPLE_COUNT sample image(s) already present."
    if ask_yes_no "Regenerate sample images anyway?" n; then
        "$VENV_PY" create_samples.py
    fi
fi

# ---- 5. Build/sync the decoupled static frontend --------------------------
# Keeps static-frontend/index.html identical to templates/index.html (plus its
# config.js include) so the two never silently drift - this is the app's only
# "build" step since the dashboard itself needs no bundler/compiler.

if [ -f "templates/index.html" ] && [ -f "static-frontend/config.js" ]; then
    info "Syncing static-frontend/index.html with templates/index.html..."
    "$VENV_PY" -c "
content = open('templates/index.html').read()
marker = '<script src=\"https://cdn.tailwindcss.com\"></script>'
if marker in content and '<script src=\"config.js\"></script>' not in content:
    content = content.replace(marker, '<script src=\"config.js\"></script>\n  ' + marker, 1)
open('static-frontend/index.html', 'w').write(content)
print('  static-frontend/index.html synced.')
"
fi

# ---- 6a. Hardware profile preview -------------------------------------------

info "Detecting local hardware profile (drives adaptive resource limits)..."
"$VENV_PY" -c "
import hardware
p = hardware.get_profile()
print(f\"  RAM: {p['total_ram_gb']}GB | CPUs: {p['cpu_count']} | Tier: {p['tier']}\")
for k, v in p['config'].items():
    print(f'    {k}: {v}')
" 2>/dev/null || warn "Could not preview hardware profile (non-fatal - the app detects this itself on startup)."

# ---- 6b. Optional Kaggle credentials -----------------------------------------

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    ok "Kaggle credentials found in environment variables - Kaggle Benchmark panel will be enabled."
elif [ -f "$KAGGLE_JSON" ]; then
    ok "Kaggle credentials found at $KAGGLE_JSON - Kaggle Benchmark panel will be enabled."
else
    warn "No Kaggle credentials found - the Kaggle Dataset Benchmark panel will show as unavailable."
    if ask_yes_no "Set up Kaggle credentials now? (from https://www.kaggle.com/settings -> API -> Create New Token)" n; then
        kaggle_user="$(ask_value "Kaggle username" "")"
        kaggle_key="$(ask_value "Kaggle API key" "")"
        if [ -n "$kaggle_user" ] && [ -n "$kaggle_key" ]; then
            mkdir -p "$HOME/.kaggle"
            printf '{"username":"%s","key":"%s"}\n' "$kaggle_user" "$kaggle_key" > "$KAGGLE_JSON"
            chmod 600 "$KAGGLE_JSON"
            ok "Saved credentials to $KAGGLE_JSON"
        else
            warn "Skipped - no username/key entered."
        fi
    else
        echo "  (You can enable this later - see the README 'Data & Credentials Required' section.)"
    fi
fi

# Re-check after any credential setup just performed above.
KAGGLE_READY=0
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    KAGGLE_READY=1
elif [ -f "$KAGGLE_JSON" ]; then
    KAGGLE_READY=1
fi

# ---- 7. Optional: pre-fetch the Kaggle benchmark dataset --------------------
# Downloads (and lets kagglehub cache) the curated dataset now, so the first
# click of "Run Benchmark" in the UI doesn't pay the download cost live.

if [ "$KAGGLE_READY" = "1" ]; then
    if ask_yes_no "Pre-fetch the Kaggle benchmark dataset now (mcagriaksoy/trees-in-satellite-imagery)?" n; then
        info "Downloading via kagglehub (cached locally afterward)..."
        if "$VENV_PY" -c "
import kagglehub
path = kagglehub.dataset_download('mcagriaksoy/trees-in-satellite-imagery')
print(f'  Cached at: {path}')
"; then
            ok "Kaggle dataset ready."
        else
            warn "Kaggle dataset download failed - check your credentials/network. The app still works without it."
        fi
    fi
fi

# ---- 8. Optional: build the Docker image locally ---------------------------
# Not needed to run the app (that's what the rest of this script does), but
# useful to verify the container build before pushing to a cloud host.

if command -v docker >/dev/null 2>&1; then
    if ask_yes_no "Docker detected. Build the container image locally too? (optional, verifies the deployable build)" n; then
        info "Building Docker image 'flora-carbon-ai:local'..."
        if docker build -t flora-carbon-ai:local .; then
            ok "Docker image built: flora-carbon-ai:local (run with: docker run -p 7860:7860 flora-carbon-ai:local)"
        else
            warn "Docker build failed - see the output above. This does not block running the app locally."
        fi
    fi
else
    warn "Docker not found on PATH - skipping the optional container build (not required to run locally)."
fi

# ---- 9. Port selection & availability check --------------------------------

DEFAULT_PORT=7860
PORT="$(ask_value "Port to run on" "$DEFAULT_PORT")"

port_in_use() {
    "$VENV_PY" -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('127.0.0.1', $1))
    s.close()
    exit(1)
except OSError:
    exit(0)
"
}

while port_in_use "$PORT"; do
    warn "Port $PORT is already in use."
    if ask_yes_no "Try a different port?" y; then
        PORT="$(ask_value "Port to run on" "$((PORT + 1))")"
    else
        fail "Cannot start - port $PORT is busy. Free it or choose another port."
        exit 1
    fi
done
ok "Port $PORT is available."

# ---- 10. Launch ---------------------------------------------------------------

OPEN_BROWSER=0
if ask_yes_no "Open the dashboard in your browser once the server is up?" y; then
    OPEN_BROWSER=1
fi

printf "\n${BOLD}Starting Flora Carbon AI on http://127.0.0.1:%s ...${RESET}\n" "$PORT"
echo "Press Ctrl+C to stop."
echo

if [ "$OPEN_BROWSER" = "1" ]; then
    (
        sleep 2
        if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:$PORT"
        elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:$PORT"
        fi
    ) &
fi

export PORT
exec "$VENV_PY" app.py
