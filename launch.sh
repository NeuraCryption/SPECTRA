#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD=""

if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
fi

if [ -z "$PYTHON_CMD" ]; then
    for cand in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/venv" "$HOME/.venv" "$HOME/venv" "$HOME/myenv"; do
        for p in "$cand/bin/python3" "$cand/bin/python"; do
            if [ -x "$p" ]; then
                if "$p" -c "import torch, transformers, fastapi" &>/dev/null; then
                    PYTHON_CMD="$p"
                    break 2
                fi
            fi
        done
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python 3 was not found. Please install Python 3.10+." >&2
        exit 1
    fi
fi

if ! $PYTHON_CMD -c "import torch, transformers, fastapi, uvicorn" &>/dev/null; then
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ ! -d ".venv" ]; then
            $PYTHON_CMD -m venv .venv
        fi
        source .venv/bin/activate
        PYTHON_CMD="python3"
    fi

    if ! $PYTHON_CMD -c "import torch, transformers, fastapi, uvicorn" &>/dev/null; then
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
fi

exec $PYTHON_CMD launch.py "$@"
