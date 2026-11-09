@echo off
rem Copyright (c) 2026 NeuraCryption Private Limited. All Rights Reserved.
rem SPECTRA 1-Command Turnkey Launcher for Windows (Command Prompt / PowerShell)

echo ========================================================================
echo   SPECTRA: Universal Model Compression ^& GPU Serving Engine
echo ========================================================================

rem 1. Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [Error] Python was not found in your PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/
    echo Make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b 1
)

rem 2. Auto-provision virtual environment
if not defined VIRTUAL_ENV (
    if not exist ".venv" (
        echo [Setup] Creating isolated virtual environment in .venv...
        python -m venv .venv
    )
    echo [Setup] Activating .venv...
    call .venv\Scripts\activate.bat
)

rem 3. Check and install core dependencies
python -c "import torch, transformers, fastapi, uvicorn" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [Setup] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

rem 4. Launch Turnkey Engine & Web Dashboard
echo [SPECTRA] Starting GPU engine and web interface...
python launch.py %*
