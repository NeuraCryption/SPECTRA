#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or higher is required to run SPECTRA.")

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def ensure_environment():
    try:
        import torch
        import transformers
        import fastapi
        import uvicorn
        return
    except ImportError:
        pass

    candidate_venvs = []
    for env_var in ["VIRTUAL_ENV", "CONDA_PREFIX"]:
        val = os.environ.get(env_var)
        if val:
            candidate_venvs.append(Path(val))

    home = Path.home()
    candidate_venvs.extend([
        ROOT / ".venv",
        ROOT / "venv",
        ROOT / "env",
        home / ".venv",
        home / "venv",
        home / "env",
        home / "myenv",
    ])

    for cand in candidate_venvs:
        py_bin = cand / "Scripts" / "python.exe" if os.name == "nt" else cand / "bin" / "python3"
        if not py_bin.exists():
            py_bin = cand / "bin" / "python"
        if py_bin.exists() and str(py_bin) != sys.executable:
            try:
                res = subprocess.run([str(py_bin), "-c", "import torch, transformers, fastapi"], capture_output=True, timeout=5)
                if res.returncode == 0:
                    if os.name == "nt":
                        sys.exit(subprocess.call([str(py_bin)] + sys.argv))
                    else:
                        os.execv(str(py_bin), [str(py_bin)] + sys.argv)
            except Exception:
                pass

    venv_dir = ROOT / ".venv"
    py_bin = venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python3"
    req_file = ROOT / "requirements.txt"

    if not py_bin.exists():
        try:
            import venv
            venv.create(str(venv_dir), with_pip=True)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            return

    subprocess.check_call([str(py_bin), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py_bin), "-m", "pip", "install", "-r", str(req_file)])

    if os.name == "nt":
        sys.exit(subprocess.call([str(py_bin)] + sys.argv))
    else:
        os.execv(str(py_bin), [str(py_bin)] + sys.argv)

if __name__ == "__main__":
    ensure_environment()
    from spectra.cli import main
    main()
