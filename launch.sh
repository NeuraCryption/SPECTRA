#!/usr/bin/env bash
# Copyright (c) 2026 NeuraCryption Private Limited. All Rights Reserved.
# SPECTRA 1-Command Turnkey Launcher for Linux / macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  SPECTRA: Universal Model Compression & GPU Serving Engine"
echo "========================================================================"

# 1. Locate Python 3.10+
if [ -n "$VIRTUAL_ENV" ] && command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif [ -f "/home/shardul/myenv/bin/python" ]; then
    PYTHON_CMD="/home/shardul/myenv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[Error] Python 3 was not found on your system."
    echo "Please install Python 3.10, 3.11, or 3.12: https://www.python.org/downloads/"
    exit 1
fi

# 2. Check if dependencies are already satisfied
if ! $PYTHON_CMD -c "import torch, transformers, fastapi, uvicorn" &>/dev/null; then
    # Auto-provision virtual environment if needed
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ ! -d ".venv" ]; then
            echo "[Setup] Creating isolated virtual environment in .venv..."
            $PYTHON_CMD -m venv .venv
        fi
        echo "[Setup] Activating .venv..."
        source .venv/bin/activate
        PYTHON_CMD="python"
    fi

    # Install missing dependencies
    if ! $PYTHON_CMD -c "import torch, transformers, fastapi, uvicorn" &>/dev/null; then
        echo "[Setup] Installing required dependencies from requirements.txt..."
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
fi

# 3. Launch Turnkey Engine & Web Dashboard
echo "[SPECTRA] Starting GPU engine and web interface..."
exec $PYTHON_CMD launch.py "$@"
