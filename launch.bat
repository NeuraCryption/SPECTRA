@echo off
setlocal

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PY_CMD=python"
    ) else (
        echo Error: Python 3 was not found in PATH. Please install Python 3.10+.
        exit /b 1
    )
)

if not defined VIRTUAL_ENV (
    if not exist ".venv" (
        %PY_CMD% -m venv .venv
    )
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
        set "PY_CMD=python"
    )
)

%PY_CMD% -c "import torch, transformers, fastapi, uvicorn" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    %PY_CMD% -m pip install --upgrade pip
    %PY_CMD% -m pip install -r requirements.txt
)

%PY_CMD% launch.py %*
