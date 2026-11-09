# Copyright (c) 2026 NeuraCryption Private Limited (neuracryption.com). All Rights Reserved.
"""
SPECTRA: Universal High-Throughput Model Compression & Serving Engine
=====================================================================
Enterprise Python SDK for 4-bit Group-32 Microscaled GPU inference.
"""

from spectra.hardware import HardwareInspector, HardwareProfile, ExecutionPlan
from spectra.compressor import SpectraCompressor
from spectra.engine import SpectraUniversalEngine, SpectraFusedLinear, GenerationResult
from spectra.orchestrator import Spectra

# Top-level functional API
serve = Spectra.serve
load = Spectra.load
compress = Spectra.compress
probe_hardware = Spectra.probe_hardware

__version__ = "2.0.0"
__all__ = [
    "Spectra",
    "serve",
    "load",
    "compress",
    "probe_hardware",
    "SpectraCompressor",
    "SpectraUniversalEngine",
    "SpectraFusedLinear",
    "HardwareInspector",
    "HardwareProfile",
    "ExecutionPlan",
    "GenerationResult",
]
