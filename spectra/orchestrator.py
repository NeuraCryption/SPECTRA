# Copyright (c) 2026 NeuraCryption Private Limited (neuracryption.com). All Rights Reserved.
"""
Model Serving and Compression Facade
====================================
Unified entrypoint for model compression and GPU inference serving.
"""

import os
import gc
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch

from spectra.hardware import HardwareInspector, HardwareProfile, ExecutionPlan
from spectra.compressor import SpectraCompressor
from spectra.engine import SpectraUniversalEngine, GenerationResult

class Spectra:
    """
    Main entrypoint for model loading and compression.
    """
    @classmethod
    def probe_hardware(cls, device_index: int = 0) -> HardwareProfile:
        return HardwareInspector.probe(device_index=device_index)

    @classmethod
    def compress(
        cls,
        model_id_or_path: Union[str, Path],
        output_container: Optional[Union[str, Path]] = None,
        group_size: int = 32,
        outlier_ratio: float = 0.01
    ) -> Dict[str, Any]:
        """
        Compresses any HuggingFace model into a .trdc container.
        """
        model_path = Path(model_id_or_path)
        if output_container is None:
            output_container = model_path.parent / f"{model_path.name}_spectra.trdc"

        compressor = SpectraCompressor(
            group_size=group_size,
            base_outlier_ratio=outlier_ratio,
            adaptive_kurtosis=True
        )
        return compressor.compress(model_id_or_path, output_container)

    @classmethod
    def serve(
        cls,
        model_or_container: Union[str, Path],
        model_dir: Optional[Union[str, Path]] = None,
        device_index: int = 0,
        auto_compress: bool = True
    ) -> SpectraUniversalEngine:
        """
        Loads and initializes model container on GPU for inference.
        Supports:
        1. Direct .trdc file paths.
        2. Cloned HuggingFace model directories containing model.trdc or *.trdc.
        3. Raw model directories with pre-compressed containers in parent or compressed_models/.
        4. Auto-compression of raw checkpoints on demand.
        """
        target_path = Path(model_or_container).resolve()

        # Case 1: Direct .trdc container file
        if target_path.suffix == ".trdc" and target_path.exists():
            return SpectraUniversalEngine.load(
                trdc_path=target_path,
                model_dir=model_dir,
                device_index=device_index
            )

        # Case 2: Directory containing a .trdc container (e.g. cloned HuggingFace repo)
        if target_path.is_dir():
            local_trdc = list(target_path.glob("model.trdc")) + list(target_path.glob("*.trdc"))
            if local_trdc:
                chosen_trdc = local_trdc[0]
                print(f"[SPECTRA] Found container inside model directory: {chosen_trdc}")
                return SpectraUniversalEngine.load(
                    trdc_path=chosen_trdc,
                    model_dir=target_path,
                    device_index=device_index
                )

        # Case 3: Raw model directory with container located in parent or compressed_models
        default_container = target_path.parent / "compressed_models" / f"{target_path.stem}_trdc.trdc"
        if not default_container.exists():
            default_container = target_path.parent / f"{target_path.stem}_spectra.trdc"
        if not default_container.exists() and (target_path.parent / "compressed_models" / f"{target_path.stem}_spectra.trdc").exists():
            default_container = target_path.parent / "compressed_models" / f"{target_path.stem}_spectra.trdc"

        if default_container.exists():
            print(f"[SPECTRA] Found container: {default_container}")
            return SpectraUniversalEngine.load(
                trdc_path=default_container,
                model_dir=target_path,
                device_index=device_index
            )

        if auto_compress:
            out_trdc = target_path / "model.trdc" if target_path.is_dir() else default_container
            print(f"[SPECTRA] Compressing {target_path.name} to {out_trdc}...")
            cls.compress(target_path, out_trdc)
            return SpectraUniversalEngine.load(
                trdc_path=out_trdc,
                model_dir=target_path,
                device_index=device_index
            )
        else:
            raise FileNotFoundError(f"Container not found and auto_compress is False for: {target_path}")

    # Enterprise Python alias: spectra.load() == spectra.serve()
    load = serve

