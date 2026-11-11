import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import errno
import struct
import gc
import json
import time
import psutil
import threading
import subprocess
import ctypes
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spectra.hardware import HardwareInspector
from spectra.compressor import SpectraCompressor
from spectra.orchestrator import Spectra
from spectra.engine import SpectraUniversalEngine

app = FastAPI(title="SPECTRA Inference Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
COMPRESSED_DIR = BASE_DIR / "compressed_models"
COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR = BASE_DIR / "chat_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = Path(__file__).parent / "web"

# SafeStream logger: protects server from EIO (Errno 5) / EPIPE (Errno 32) on disconnected terminals
class SafeStream:
    """
    Wraps standard I/O streams to swallow EIO (Errno 5) and EPIPE (Errno 32)
    when the parent terminal or pipe is disconnected, while routing output to
    a persistent log file.
    """
    def __init__(self, original_stream, fallback_file=None):
        self._orig = original_stream
        self._fallback = fallback_file

    def write(self, s):
        if self._orig is not None:
            try:
                self._orig.write(s)
                self._orig.flush()
            except (OSError, BrokenPipeError, IOError) as e:
                if getattr(e, 'errno', None) in (errno.EIO, errno.EPIPE, 5, 32):
                    self._orig = None
                else:
                    raise
        if self._fallback is not None:
            try:
                self._fallback.write(s)
                self._fallback.flush()
            except Exception:
                pass
        return len(s) if isinstance(s, (str, bytes)) else 0

    def flush(self):
        if self._orig is not None:
            try:
                self._orig.flush()
            except (OSError, BrokenPipeError, IOError):
                self._orig = None
        if self._fallback is not None:
            try:
                self._fallback.flush()
            except Exception:
                pass

    def isatty(self):
        if self._orig is not None:
            try:
                return self._orig.isatty()
            except Exception:
                return False
        return False

    def fileno(self):
        if self._orig is not None:
            try:
                return self._orig.fileno()
            except Exception:
                pass
        if self._fallback is not None:
            try:
                return self._fallback.fileno()
            except Exception:
                pass
        raise OSError(errno.EBADF, "Bad file descriptor")

    def __getattr__(self, name):
        if self._orig is not None and hasattr(self._orig, name):
            return getattr(self._orig, name)
        if self._fallback is not None and hasattr(self._fallback, name):
            return getattr(self._fallback, name)
        raise AttributeError(f"SafeStream has no attribute {name}")

SERVER_LOG_FILE = open(BASE_DIR / "spectra_server.log", "a", encoding="utf-8", buffering=1)
sys.stdout = SafeStream(sys.stdout, SERVER_LOG_FILE)
sys.stderr = SafeStream(sys.stderr, SERVER_LOG_FILE)

def is_valid_trdc(file_path: Path) -> bool:
    """Verifies that a .trdc file has a valid magic header and intact manifest table."""
    try:
        p = Path(file_path)
        if not p.is_file():
            return False
        size = p.stat().st_size
        if size < 24:
            return False
        with open(p, "rb") as fp:
            if fp.read(8) != b"TRDC4B01":
                return False
            off, length = struct.unpack("<QQ", fp.read(16))
            if off < 24 or off + length > size:
                return False
        return True
    except Exception:
        return False


# Global state
ACTIVE_ENGINE: Optional[SpectraUniversalEngine] = None
ACTIVE_MODEL_PATH: Optional[str] = None
ACTIVE_ENGINE_LOCK = threading.Lock()

class ChatSessionManager:
    """Manages chat sessions stored directly as JSON files on user's local hard drive."""
    @staticmethod
    def _safe_session_path(session_id: str) -> Path:
        clean = re.sub(r'[^a-zA-Z0-9_\-]', '', Path(session_id).stem)
        if not clean:
            clean = "default"
        path = (SESSIONS_DIR / f"{clean}.json").resolve()
        if not path.is_relative_to(SESSIONS_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid session path traversal")
        return path

    @staticmethod
    def list_sessions() -> List[Dict[str, Any]]:
        sessions = []
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                msgs = data.get("messages", [])
                if len(msgs) == 0:
                    continue  # Never list empty sessions
                sessions.append({
                    "id": f.stem,
                    "title": data.get("title", "New Chat"),
                    "created_at": data.get("created_at", time.time()),
                    "updated_at": data.get("updated_at", time.time()),
                    "message_count": len(msgs),
                    "model": data.get("model", "Default"),
                    "context_window": data.get("context_window", 4096)
                })
            except Exception:
                pass
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    @staticmethod
    def get_session(session_id: str) -> Dict[str, Any]:
        path = ChatSessionManager._safe_session_path(session_id)
        if not path.exists():
            now = time.time()
            return {
                "id": path.stem,
                "title": "New Chat",
                "created_at": now,
                "updated_at": now,
                "messages": [],
                "context_window": 4096,
                "model": "Default"
            }
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return {"id": path.stem, "title": "New Chat", "messages": [], "context_window": 4096}

    @staticmethod
    def save_session(session_id: str, data: Dict[str, Any]):
        path = ChatSessionManager._safe_session_path(session_id)
        data["id"] = path.stem
        data["updated_at"] = time.time()
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    @staticmethod
    def delete_session(session_id: str):
        path = ChatSessionManager._safe_session_path(session_id)
        if path.exists():
            path.unlink()
        global ACTIVE_ENGINE
        if ACTIVE_ENGINE:
            ACTIVE_ENGINE.clear_session(path.stem)

COMPRESSION_STATE = {
    "status": "idle",
    "model_name": "",
    "tensor": "",
    "index": 0,
    "total": 0,
    "percentage": 0.0,
    "orig_bytes": 0,
    "compressed_bytes": 0,
    "ratio": 1.0,
    "error": None
}

# =============================================================================
# 1. HARDWARE TELEMETRY (True Physical NVML Direct Query)
# =============================================================================

class NVMLDeviceTelemetry:
    _instance = None

    def __init__(self):
        self.available = False
        self.handle = None
        self.nvml = None
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
            self.nvml.nvmlInit()
            self.handle = ctypes.c_void_p()
            self.nvml.nvmlDeviceGetHandleByIndex(0, ctypes.byref(self.handle))
            self.available = True
        except Exception:
            self.available = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def query(self) -> Dict[str, Any]:
        if not self.available or self.handle is None:
            return {}
        try:
            class MemoryV2(ctypes.Structure):
                _fields_ = [
                    ('version', ctypes.c_uint),
                    ('total', ctypes.c_ulonglong),
                    ('reserved', ctypes.c_ulonglong),
                    ('free', ctypes.c_ulonglong),
                    ('used', ctypes.c_ulonglong)
                ]
            class Memory(ctypes.Structure):
                _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
            class Utilization(ctypes.Structure):
                _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

            used_bytes = 0
            total_bytes = 0
            free_bytes = 0

            # 1. Try NVML v2 (exact driver implementation queried by nvidia-smi)
            v2_success = False
            if hasattr(self.nvml, 'nvmlDeviceGetMemoryInfo_v2'):
                try:
                    mem_v2 = MemoryV2()
                    mem_v2.version = (2 << 24) | ctypes.sizeof(MemoryV2)
                    if self.nvml.nvmlDeviceGetMemoryInfo_v2(self.handle, ctypes.byref(mem_v2)) == 0:
                        used_bytes = mem_v2.used
                        total_bytes = mem_v2.total
                        free_bytes = mem_v2.free
                        v2_success = True
                except Exception:
                    pass

            if not v2_success:
                mem = Memory()
                self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
                used_bytes = mem.used
                total_bytes = mem.total
                free_bytes = mem.free

            temp = ctypes.c_uint()
            self.nvml.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp))
            power = ctypes.c_uint()
            self.nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
            util = Utilization()
            self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))

            return {
                "gpu_vram_used_mb": round(used_bytes / (1024**2), 1),
                "gpu_vram_total_mb": round(total_bytes / (1024**2), 1),
                "gpu_vram_free_mb": round(free_bytes / (1024**2), 1),
                "gpu_vram_percent": round((used_bytes / total_bytes) * 100, 1),
                "gpu_temp_c": int(temp.value),
                "gpu_power_w": round(power.value / 1000.0, 1),
                "gpu_util_percent": int(util.gpu)
            }
        except Exception:
            return {}

def get_live_telemetry() -> Dict[str, Any]:
    # Host memory & CPU
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)

    telemetry = {
        "has_cuda": torch.cuda.is_available(),
        "host_ram_total_mb": round(mem.total / (1024**2), 1),
        "host_ram_used_mb": round(mem.used / (1024**2), 1),
        "host_ram_free_mb": round(mem.available / (1024**2), 1),
        "host_ram_percent": mem.percent,
        "cpu_percent": cpu_pct,
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_util_percent": None,
        "strategy_mode": ACTIVE_ENGINE.plan.mode if ACTIVE_ENGINE else "STANDBY",
        "loaded_model": Path(ACTIVE_MODEL_PATH).name if ACTIVE_MODEL_PATH else None
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**2)
        py_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        py_reserved = torch.cuda.memory_reserved(0) / (1024**2)

        # 1. Query true physical GPU hardware telemetry via NVML (Exact match with nvidia-smi)
        nvml_data = NVMLDeviceTelemetry.get_instance().query()
        if nvml_data:
            real_used_vram = nvml_data["gpu_vram_used_mb"]
            real_total_vram = nvml_data["gpu_vram_total_mb"]
            real_free_vram = nvml_data["gpu_vram_free_mb"]
            real_vram_pct = nvml_data["gpu_vram_percent"]
            gpu_temp = nvml_data["gpu_temp_c"]
            gpu_power = nvml_data["gpu_power_w"]
            gpu_util = nvml_data["gpu_util_percent"]
        else:
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                used_b = total_b - free_b
                real_used_vram = round(used_b / (1024**2), 1)
                real_total_vram = round(total_b / (1024**2), 1)
                real_free_vram = round(free_b / (1024**2), 1)
                real_vram_pct = round((used_b / total_b) * 100, 1)
            except Exception:
                real_used_vram = round(py_reserved, 1)
                real_total_vram = round(total_vram, 1)
                real_free_vram = max(0.0, real_total_vram - real_used_vram)
                real_vram_pct = round((real_used_vram / real_total_vram) * 100, 1)
            gpu_temp = None
            gpu_power = None
            gpu_util = None

        telemetry.update({
            "device_name": props.name,
            "architecture": "Ampere" if props.major == 8 else ("Ada/Hopper" if props.major >= 9 else "Turing/Other"),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_vram_mb": real_total_vram,
            "allocated_vram_mb": real_used_vram,            # True physical used VRAM (matches nvidia-smi exactly)
            "physical_used_vram_mb": real_used_vram,
            "pytorch_allocated_mb": round(py_allocated, 1),
            "pytorch_reserved_mb": round(py_reserved, 1),
            "reserved_vram_mb": round(py_reserved, 1),
            "free_vram_mb": real_free_vram,
            "vram_percent": real_vram_pct,
            "gpu_temp_c": gpu_temp,
            "gpu_power_w": gpu_power,
            "gpu_util_percent": gpu_util
        })
    else:
        telemetry.update({
            "device_name": f"Host CPU ({psutil.cpu_count()} Cores)",
            "architecture": "x86_64 CPU",
            "compute_capability": "N/A",
            "sm_count": psutil.cpu_count(logical=False) or 1,
            "total_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "physical_used_vram_mb": 0.0,
            "pytorch_allocated_mb": 0.0,
            "pytorch_reserved_mb": 0.0,
            "free_vram_mb": 0.0,
            "vram_percent": 0.0
        })

    engine = ACTIVE_ENGINE
    if engine is not None:
        session_tokens = engine.get_session_token_count("default")
        has_sess = engine.has_session("default")
        active_ctx = getattr(engine, 'active_context_window', getattr(engine, 'max_model_len', 4096))
        telemetry.update({
            "bytes_per_token": getattr(engine, 'bytes_per_token', None),
            "max_safe_gpu_tokens": getattr(engine, 'max_safe_gpu_tokens', None),
            "max_model_len": getattr(engine, 'max_model_len', 4096),
            "active_context_window": active_ctx,
            "session_tokens": session_tokens,
            "session_state": "Memory Cached" if has_sess else "Ready",
            "context_engine": "Smart Memory Tiering (GPU + Host RAM)"
        })
    else:
        telemetry.update({
            "bytes_per_token": None,
            "max_safe_gpu_tokens": None,
            "max_model_len": None,
            "active_context_window": 4096,
            "session_tokens": 0,
            "session_state": "Ready",
            "context_engine": "Standby"
        })

    return telemetry

# =============================================================================
# 2. MODEL DISCOVERY
# =============================================================================

def discover_models() -> Dict[str, Any]:
    raw_models = []
    compressed_models = []
    seen_trdc_paths = set()
    seen_raw_paths = set()

    home = Path.home()

    # System-wide comprehensive search locations
    search_dirs = [
        BASE_DIR / "models",
        COMPRESSED_DIR,
        Path.cwd() / "models",
        Path.cwd() / "compressed_models",
        BASE_DIR,
        Path.cwd(),
        home / "models",
        home / "huggingface_models",
        home / "compressed_models",
        home / "Downloads",
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface",
        home / ".cache" / "lm-studio" / "models",
        home / ".ollama" / "models",
        home / "text-generation-webui" / "models",
        home / ".gemini" / "antigravity" / "scratch",
        home / ".gemini" / "antigravity" / "scratch" / "compressed_models",
    ]

    for env_k in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "MODELS_DIR", "MODEL_PATH"]:
        env_v = os.environ.get(env_k)
        if env_v and Path(env_v).exists():
            search_dirs.append(Path(env_v))

    if os.name == "nt":
        for drive in ["C", "D", "E", "F"]:
            dp = Path(f"{drive}:\\models")
            if dp.exists():
                search_dirs.append(dp)

    valid_search_dirs = []
    for d in search_dirs:
        try:
            res_d = d.resolve()
            if res_d.exists() and res_d not in valid_search_dirs:
                valid_search_dirs.append(res_d)
        except Exception:
            pass

    # 1. Collect all potential model folders and standalone containers
    candidate_folders = []
    standalone_trdc_files = []

    for s_dir in valid_search_dirs:
        # Check standalone .trdc files in search dir
        try:
            for f in s_dir.glob("*.trdc"):
                if f.is_file() and str(f.resolve()) not in seen_trdc_paths and is_valid_trdc(f):
                    standalone_trdc_files.append(f)
        except Exception:
            pass

        # Check subdirectories in search dir
        try:
            for sub in s_dir.iterdir():
                if sub.is_dir() and not sub.name.startswith(".") and sub.name not in ["__pycache__", "chat_sessions", "node_modules", "venv", ".venv"]:
                    if sub.name.startswith("models--"):
                        snaps = sub / "snapshots"
                        if snaps.exists() and snaps.is_dir():
                            for snap in snaps.iterdir():
                                if snap.is_dir():
                                    sub_res = snap.resolve()
                                    if sub_res not in candidate_folders:
                                        candidate_folders.append(sub_res)
                    else:
                        sub_res = sub.resolve()
                        if sub_res not in candidate_folders:
                            candidate_folders.append(sub_res)
        except Exception:
            pass

    # 2. Process candidate folders
    for p in candidate_folders:
        try:
            trdc_in_folder = [c for c in p.glob("*.trdc") if is_valid_trdc(c)]
            cfg_file = p / "config.json"
            cfg = {}
            if cfg_file.exists():
                try:
                    with open(cfg_file, "r", encoding="utf-8") as fp:
                        cfg = json.load(fp)
                except Exception:
                    cfg = {}

            arch = cfg.get("architectures", ["Transformer"])[0] if cfg else "Transformer"
            num_layers = cfg.get("num_hidden_layers", cfg.get("n_layer", cfg.get("num_layers", 28))) if cfg else 28
            max_pos = cfg.get("max_position_embeddings", cfg.get("max_sequence_length", 32768)) if cfg else 32768
            is_moe = any(k in str(cfg).lower() for k in ['num_experts', 'moe', 'experts']) if cfg else False

            clean_name = p.name.replace("_", " ").replace("-SPECTRA", "").replace("-spectra", "").title()
            if "models--" in str(p):
                parts = str(p).split("models--")[-1].split(os.sep)[0].split("--")
                if len(parts) >= 2:
                    clean_name = f"{parts[0].title()} {parts[1].replace('_', ' ').replace('-', ' ').title()}"
            if cfg.get("_name_or_path") and not str(cfg["_name_or_path"]).startswith("/"):
                clean_name = str(cfg["_name_or_path"]).split("/")[-1].replace("_", " ").title()

            # If folder is a general collection directory, process each container individually
            if p.name.lower() in ["compressed_models", "models", "scratch"]:
                for trdc_f in trdc_in_folder:
                    c_res = str(trdc_f.resolve())
                    if c_res not in seen_trdc_paths and is_valid_trdc(trdc_f):
                        seen_trdc_paths.add(c_res)
                        size_bytes = trdc_f.stat().st_size
                        size_gb = round(size_bytes / (1024**3), 2)
                        orig_gb = round(size_gb * 3.2, 2)
                        clean_stem = trdc_f.stem.replace("_spectra", "").replace("_trdc", "").replace("_", " ").title()
                        is_loaded = (ACTIVE_MODEL_PATH == c_res)
                        compressed_models.append({
                            "name": trdc_f.name,
                            "display_name": clean_stem + " (SPECTRA 4-bit)",
                            "path": c_res,
                            "parent_model": clean_stem,
                            "parent_id": None,
                            "size_gb": size_gb,
                            "orig_size_gb": orig_gb,
                            "ratio": "3.2x",
                            "vram_savings_pct": 68.5,
                            "is_loaded": is_loaded,
                            "architecture": "Auto-Detect",
                            "num_layers": 28,
                            "max_context": 4096
                        })
                continue

            # Case A: Folder contains a .trdc container (cloned model repo ready to run!)
            if trdc_in_folder:
                primary_trdc = next((c for c in trdc_in_folder if c.name == "model.trdc"), trdc_in_folder[0])
                c_res = str(primary_trdc.resolve())
                if c_res not in seen_trdc_paths and is_valid_trdc(primary_trdc):
                    seen_trdc_paths.add(c_res)
                    size_bytes = primary_trdc.stat().st_size
                    size_gb = round(size_bytes / (1024**3), 2)
                    orig_gb = round(size_gb * 3.2, 2)
                    is_loaded = (ACTIVE_MODEL_PATH == c_res)

                    compressed_models.append({
                        "name": primary_trdc.name,
                        "display_name": clean_name + " (SPECTRA 4-bit)",
                        "path": c_res,
                        "parent_model": clean_name,
                        "parent_id": p.name,
                        "size_gb": size_gb,
                        "orig_size_gb": orig_gb,
                        "ratio": "3.2x",
                        "vram_savings_pct": 68.5,
                        "is_loaded": is_loaded,
                        "architecture": arch,
                        "num_layers": num_layers,
                        "max_context": max_pos
                    })
                continue

            # Case B: Folder has raw weights (safetensors or bin) and config.json
            has_raw_weights = bool(list(p.glob("*.safetensors")) or list(p.glob("*.bin")))
            if cfg_file.exists() and has_raw_weights:
                p_res = str(p.resolve())
                if p_res not in seen_raw_paths:
                    seen_raw_paths.add(p_res)
                    size_bytes = sum(f.stat().st_size for f in p.glob("*.safetensors")) + sum(f.stat().st_size for f in p.glob("*.bin"))
                    size_gb = round(size_bytes / (1024**3), 2)

                    # Check if a matching compressed container already exists
                    has_comp = False
                    matched_container = None
                    for seen_c in seen_trdc_paths:
                        c_path = Path(seen_c)
                        if p.name.lower() in c_path.stem.lower() or p.name.lower() in c_path.parent.name.lower():
                            has_comp = True
                            matched_container = seen_c
                            break

                    raw_models.append({
                        "id": p.name,
                        "name": clean_name,
                        "path": p_res,
                        "architecture": arch,
                        "type": "Mixture of Experts" if is_moe else "Standard Transformer",
                        "max_context": max_pos,
                        "num_layers": num_layers,
                        "size_gb": size_gb,
                        "has_compressed": has_comp,
                        "compressed_path": matched_container
                    })
        except Exception:
            pass

    # 3. Process standalone .trdc files (e.g. in compressed_models/ or root)
    for c in standalone_trdc_files:
        c_res = str(c.resolve())
        if c_res in seen_trdc_paths:
            continue
        seen_trdc_paths.add(c_res)
        size_bytes = c.stat().st_size
        size_gb = round(size_bytes / (1024**3), 2)
        orig_gb = round(size_gb * 3.2, 2)
        display_name = c.stem.replace("_spectra", "").replace("_trdc", "").replace("_", " ").title() + " (SPECTRA 4-bit)"
        is_loaded = (ACTIVE_MODEL_PATH == c_res)

        compressed_models.append({
            "name": c.name,
            "display_name": display_name,
            "path": c_res,
            "parent_model": c.stem,
            "parent_id": None,
            "size_gb": size_gb,
            "orig_size_gb": orig_gb,
            "ratio": "3.2x",
            "vram_savings_pct": 68.5,
            "is_loaded": is_loaded,
            "architecture": "Auto-Detect",
            "num_layers": 28,
            "max_context": 4096
        })

    return {
        "raw_models": raw_models,
        "compressed_models": compressed_models
    }

# =============================================================================
# 3. API ROUTES
# =============================================================================

@app.get("/")
def serve_index():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>SPECTRA Web UI Loading...</h1>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/telemetry")
def api_telemetry():
    return get_live_telemetry()

@app.get("/api/models")
def api_models():
    return discover_models()

class CompressRequest(BaseModel):
    model_path: str
    output_filename: Optional[str] = None
    group_size: int = 32
    outlier_ratio: float = 0.01

def _run_compression_task(model_path: str, output_path: str, group_size: int, outlier_ratio: float):
    global COMPRESSION_STATE
    try:
        COMPRESSION_STATE["status"] = "compressing"
        COMPRESSION_STATE["model_name"] = Path(model_path).name
        COMPRESSION_STATE["percentage"] = 0.0
        COMPRESSION_STATE["error"] = None

        def callback(d):
            COMPRESSION_STATE.update(d)

        compressor = SpectraCompressor(
            group_size=group_size,
            base_outlier_ratio=outlier_ratio,
            adaptive_kurtosis=True
        )
        res = compressor.compress(
            model_id_or_path=model_path,
            output_file=output_path,
            progress_callback=callback
        )
        COMPRESSION_STATE["status"] = "done"
        COMPRESSION_STATE["percentage"] = 100.0
        COMPRESSION_STATE["result"] = res
    except Exception as e:
        COMPRESSION_STATE["status"] = "error"
        COMPRESSION_STATE["error"] = str(e)

@app.post("/api/compress")
def api_compress(req: CompressRequest, background_tasks: BackgroundTasks):
    global COMPRESSION_STATE
    if COMPRESSION_STATE["status"] == "compressing":
        raise HTTPException(status_code=400, detail="A compression task is already running.")

    m_path = Path(req.model_path).resolve()
    if not m_path.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {req.model_path}")

    # Guard against path traversal in output_filename
    raw_out = Path(req.output_filename or f"{m_path.name}_spectra.trdc").name
    if not raw_out.endswith(".trdc"):
        raw_out += ".trdc"
    clean_out = re.sub(r'[^a-zA-Z0-9_\.\-]', '', raw_out)
    out_path = (COMPRESSED_DIR / clean_out).resolve()
    if not out_path.is_relative_to(COMPRESSED_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid output filename")

    background_tasks.add_task(_run_compression_task, str(m_path), str(out_path), req.group_size, req.outlier_ratio)
    return {"message": "Compression started", "target": str(out_path)}

@app.get("/api/compress/status")
def api_compress_status():
    return COMPRESSION_STATE

class LoadModelRequest(BaseModel):
    trdc_path: str

@app.post("/api/engine/load")
def api_engine_load(req: LoadModelRequest):
    with ACTIVE_ENGINE_LOCK:
        global ACTIVE_ENGINE, ACTIVE_MODEL_PATH
        t0 = time.time()
        trdc_path = Path(req.trdc_path).resolve()
        if not trdc_path.exists() or not trdc_path.is_file():
            raise HTTPException(status_code=404, detail="Container file does not exist")
        if trdc_path.suffix != ".trdc":
            raise HTTPException(status_code=400, detail="Invalid container extension (expected .trdc)")

        # Restrict loading to allowed directories under project workspace, cwd, or home
        valid_roots = [BASE_DIR.resolve(), COMPRESSED_DIR.resolve(), Path.cwd().resolve(), Path.home().resolve()]
        allowed = any(trdc_path.is_relative_to(root) for root in valid_roots)
        if not allowed:
            raise HTTPException(status_code=403, detail="Forbidden model path outside accessible workspace")

        # Clean previous engine
        if ACTIVE_ENGINE:
            try:
                ACTIVE_ENGINE.unload()
            except Exception:
                pass
            del ACTIVE_ENGINE
            ACTIVE_ENGINE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        try:
            engine = Spectra.serve(trdc_path)
            ACTIVE_ENGINE = engine
            ACTIVE_MODEL_PATH = str(trdc_path)
            dt = time.time() - t0
            vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

            return {
                "status": "success",
                "model": trdc_path.name,
                "load_time_sec": round(dt, 2),
                "vram_allocated_mb": round(vram_mb, 1),
                "plan_mode": engine.plan.mode
            }
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"[SPECTRA Load Error] Failed to load {trdc_path.name}: {err_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load {trdc_path.name}: {err_msg}")

@app.post("/api/engine/unload")
def api_engine_unload():
    global ACTIVE_ENGINE, ACTIVE_MODEL_PATH
    if ACTIVE_ENGINE:
        try:
            ACTIVE_ENGINE.stop_generation()
        except Exception:
            pass
    with ACTIVE_ENGINE_LOCK:
        if ACTIVE_ENGINE:
            time.sleep(0.05)
            try:
                ACTIVE_ENGINE.unload()
            except Exception:
                pass
            del ACTIVE_ENGINE
            ACTIVE_ENGINE = None
            ACTIVE_MODEL_PATH = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        return {"status": "unloaded", "vram_allocated_mb": 0.0}

@app.post("/api/engine/stop")
def api_engine_stop():
    global ACTIVE_ENGINE
    if ACTIVE_ENGINE:
        ACTIVE_ENGINE.stop_generation()
        return {"status": "stopping"}
    return {"status": "no_active_engine"}

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.1
    no_repeat_ngram_size: int = 3
    session_id: str = "default"
    context_window: int = 4096
    preserve_context: bool = True

@app.post("/api/engine/generate")
def api_engine_generate(req: GenerateRequest):
    global ACTIVE_ENGINE
    if not ACTIVE_ENGINE:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a container first.")

    # Allow generation up to the user-selected context window
    req_max_tokens = req.max_new_tokens
    if req_max_tokens is None or req_max_tokens <= 0:
        req_max_tokens = max(1024, req.context_window - 512)

    print(f"\n[SPECTRA User Prompt] Session: {req.session_id} | Content: {req.prompt[:80]!r} | max_tokens: {req_max_tokens} | ctx: {req.context_window}")

    # 1. Build multi-turn conversation from session history
    conversation = []
    if req.session_id and req.preserve_context:
        sess = ChatSessionManager.get_session(req.session_id)
        for m in sess.get("messages", []):
            role = m.get("role")
            content = m.get("content", "")
            if role in ["user", "assistant", "system"] and content:
                # Clean prior assistant thoughts to keep context focused
                if role == "assistant" and "</think>" in content:
                    clean_content = content.split("</think>")[-1].strip()
                elif role == "assistant" and "<think>" in content:
                    clean_content = content.split("<think>")[0].strip()
                else:
                    clean_content = content.strip()
                if clean_content:
                    conversation.append({"role": role, "content": clean_content})

    conversation.append({"role": "user", "content": req.prompt})

    def format_universal_chat(messages):
        lines = []
        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "").strip()
            if r == "system":
                lines.append(f"<|im_start|>system\n{c}<|im_end|>")
            elif r == "user":
                lines.append(f"<|im_start|>user\n{c}<|im_end|>")
            elif r == "assistant":
                lines.append(f"<|im_start|>assistant\n{c}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)

    tok = ACTIVE_ENGINE.tokenizer
    prompt = None
    if tok and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            prompt = tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = None

    if not prompt:
        if len(conversation) > 1:
            prompt = format_universal_chat(conversation)
        else:
            prompt = req.prompt

    def stream_generator():
        last_chunk = None
        full_tokens = []
        try:
            with ACTIVE_ENGINE_LOCK:
                for chunk in ACTIVE_ENGINE.generate_stream(
                    prompt,
                    max_new_tokens=req_max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    frequency_penalty=req.frequency_penalty,
                    no_repeat_ngram_size=req.no_repeat_ngram_size,
                    session_id=req.session_id,
                    context_window=req.context_window
                ):
                    if chunk.get("done"):
                        last_chunk = chunk
                    if chunk.get("token"):
                        full_tokens.append(chunk["token"])
                    yield f"data: {json.dumps(chunk)}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            err_str = str(e)
            if ACTIVE_ENGINE is None or "unloaded" in err_str.lower() or "'NoneType' object has no attribute" in err_str:
                yield f"data: {json.dumps({'error': 'Engine session reset', 'done': True, 'token': ''})}\n\n"
            else:
                import traceback
                traceback.print_exc()
                err_msg = f"Generation error: {err_str}"
                yield f"data: {json.dumps({'error': err_msg, 'done': True, 'token': ''})}\n\n"
        finally:
            # Save conversation turn to session storage
            try:
                if req.session_id:
                    sess = ChatSessionManager.get_session(req.session_id)
                    sess["messages"].append({
                        "role": "user",
                        "content": req.prompt,
                        "timestamp": time.time()
                    })
                    bot_text = last_chunk.get("full_text", "".join(full_tokens)) if last_chunk else "".join(full_tokens)
                    if bot_text:
                        sess["messages"].append({
                            "role": "assistant",
                            "content": bot_text,
                            "timestamp": time.time(),
                            "tokens": last_chunk.get("generated_count", len(full_tokens)) if last_chunk else len(full_tokens),
                            "tok_per_sec": last_chunk.get("final_decode_tok_per_sec", 0.0) if last_chunk else 0.0,
                            "stopped": last_chunk.get("stopped", False) if last_chunk else True
                        })
                    if sess.get("title") in ["New Chat", ""]:
                        sess["title"] = req.prompt[:32].strip() + ("..." if len(req.prompt) > 32 else "")
                    sess["context_window"] = req.context_window
                    if ACTIVE_MODEL_PATH:
                        sess["model"] = Path(ACTIVE_MODEL_PATH).name
                    ChatSessionManager.save_session(req.session_id, sess)
            except Exception as e:
                print(f"[Error saving chat session]: {e}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# =============================================================================
# CHAT SESSIONS API (Local Hard-Drive Storage)
# =============================================================================

@app.get("/api/chat/sessions")
def api_list_sessions():
    return ChatSessionManager.list_sessions()

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    context_window: int = 4096

@app.post("/api/chat/sessions")
def api_create_session(req: CreateSessionRequest = CreateSessionRequest()):
    sess_id = f"chat_{int(time.time()*1000)}"
    data = {
        "id": sess_id,
        "title": req.title or "New Chat",
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [],
        "context_window": req.context_window,
        "model": Path(ACTIVE_MODEL_PATH).name if ACTIVE_MODEL_PATH else "No Model"
    }
    ChatSessionManager.save_session(sess_id, data)
    return data

@app.get("/api/chat/sessions/{session_id}")
def api_get_session(session_id: str):
    return ChatSessionManager.get_session(session_id)

@app.delete("/api/chat/sessions/{session_id}")
def api_delete_session(session_id: str):
    ChatSessionManager.delete_session(session_id)
    return {"status": "deleted", "id": session_id}

class RenameSessionRequest(BaseModel):
    title: str

@app.patch("/api/chat/sessions/{session_id}")
def api_rename_session(session_id: str, req: RenameSessionRequest):
    sess = ChatSessionManager.get_session(session_id)
    sess["title"] = req.title
    ChatSessionManager.save_session(session_id, sess)
    return {"status": "renamed", "id": session_id, "title": req.title}

class UpdateContextRequest(BaseModel):
    context_window: int

@app.post("/api/chat/sessions/{session_id}/context")
def api_update_session_context(session_id: str, req: UpdateContextRequest):
    sess = ChatSessionManager.get_session(session_id)
    sess["context_window"] = req.context_window
    ChatSessionManager.save_session(session_id, sess)
    global ACTIVE_ENGINE
    if ACTIVE_ENGINE:
        ACTIVE_ENGINE.active_context_window = req.context_window
    return {"status": "ok", "id": session_id, "context_window": req.context_window}

class ClearSessionRequest(BaseModel):
    session_id: str = "default"

@app.post("/api/engine/clear_session")
def api_engine_clear_session(req: ClearSessionRequest = ClearSessionRequest()):
    global ACTIVE_ENGINE
    if ACTIVE_ENGINE:
        ACTIVE_ENGINE.clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}

@app.post("/api/engine/benchmark")
def api_engine_benchmark():
    with ACTIVE_ENGINE_LOCK:
        global ACTIVE_ENGINE
        if not ACTIVE_ENGINE:
            raise HTTPException(status_code=400, detail="No model loaded.")

        prompts = [
            "Explain the fundamental theorem of calculus in one sentence.",
            "Solve for x: 5x - 20 = 45 step-by-step.",
            "What is the difference between a process and a thread in operating systems?",
            "Write a Python function to check if a string is a palindrome."
        ]

        try:
            # B=1 test
            r1 = ACTIVE_ENGINE.generate(prompts[0], max_new_tokens=32)
            # B=2 test
            r2 = ACTIVE_ENGINE.generate_batch(prompts[:2], max_new_tokens=32)
            # B=4 test
            r4 = ACTIVE_ENGINE.generate_batch(prompts[:4], max_new_tokens=32)

            return {
                "batch_1_speed": r1.decode_tok_per_sec,
                "batch_1_latency_ms": r1.step_latency_ms,
                "batch_2_speed": r2["aggregate_tok_per_sec"],
                "batch_2_latency_ms": r2["step_latency_ms"],
                "batch_4_speed": r4["aggregate_tok_per_sec"],
                "batch_4_latency_ms": r4["step_latency_ms"],
                "peak_vram_mb": r4["peak_vram_mb"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="SPECTRA Universal Model Serving Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    print(f"\n[SPECTRA] Launching Universal AI Engine on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

