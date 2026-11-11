import os
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch

@dataclass
class HardwareProfile:
    device_type: str
    device_index: int
    device_name: str
    compute_capability: Tuple[int, int]
    sm_count: int
    total_vram_mb: float
    free_vram_mb: float
    host_ram_total_mb: float
    host_ram_free_mb: float
    supports_tensor_cores: bool
    supports_bf16: bool
    supports_cuda_graphs: bool
    tensor_core_arch: str

    def summary_card(self) -> str:
        return (
            f"[Hardware] Device: {self.device_name} ({self.tensor_core_arch}) | "
            f"VRAM: {self.total_vram_mb:,.1f} MB (Usable: {self.free_vram_mb:,.1f} MB) | "
            f"RAM: {self.host_ram_total_mb:,.1f} MB"
        )


@dataclass
class ExecutionPlan:
    mode: str
    gpu_layers: int
    cpu_layers: int
    enable_cuda_graphs: bool
    target_device: str
    kv_cache_budget_mb: float
    safety_margin_mb: float
    chunk_size_prefill: int
    recommended_batch_size: int

    def plan_card(self) -> str:
        return (
            f"[Execution Plan] Mode: {self.mode} | "
            f"GPU Layers: {self.gpu_layers} | Host Layers: {self.cpu_layers} | "
            f"KV Budget: {self.kv_cache_budget_mb:,.1f} MB"
        )


class HardwareInspector:
    SAFETY_HEADROOM_MB = 450.0

    @classmethod
    def probe(cls, device_index: int = 0) -> HardwareProfile:
        # 1. Probe Host RAM
        mem = psutil.virtual_memory()
        host_ram_total = mem.total / (1024**2)
        host_ram_free = mem.available / (1024**2)

        # 2. Probe CUDA Devices
        has_cuda = torch.cuda.is_available() and (torch.cuda.device_count() > device_index)

        if has_cuda:
            props = torch.cuda.get_device_properties(device_index)
            dev_name = props.name
            cc = (props.major, props.minor)
            sm_count = props.multi_processor_count
            total_vram = props.total_memory / (1024**2)

            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device_index) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_index) / (1024**2)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                free_vram = free_bytes / (1024**2)
            except Exception:
                free_vram = max(0.0, total_vram - reserved)

            supports_tc = cc[0] >= 7
            supports_bf16 = cc[0] >= 8
            supports_graphs = True

            # Architecture classification
            if cc[0] >= 9:
                tc_arch = "Hopper / Blackwell"
            elif cc[0] == 8 and cc[1] >= 9:
                tc_arch = "Ada Lovelace"
            elif cc[0] == 8:
                tc_arch = "Ampere"
            elif cc[0] == 7 and cc[1] == 5:
                tc_arch = "Turing"
            elif cc[0] == 7:
                tc_arch = "Volta"
            else:
                tc_arch = "Pascal / Legacy"

            return HardwareProfile(
                device_type="cuda",
                device_index=device_index,
                device_name=dev_name,
                compute_capability=cc,
                sm_count=sm_count,
                total_vram_mb=total_vram,
                free_vram_mb=free_vram,
                host_ram_total_mb=host_ram_total,
                host_ram_free_mb=host_ram_free,
                supports_tensor_cores=supports_tc,
                supports_bf16=supports_bf16,
                supports_cuda_graphs=supports_graphs,
                tensor_core_arch=tc_arch
            )
        else:
            return HardwareProfile(
                device_type="cpu",
                device_index=-1,
                device_name=f"Host CPU ({psutil.cpu_count(logical=True)} Threads)",
                compute_capability=(0, 0),
                sm_count=psutil.cpu_count(logical=False) or 1,
                total_vram_mb=0.0,
                free_vram_mb=0.0,
                host_ram_total_mb=host_ram_total,
                host_ram_free_mb=host_ram_free,
                supports_tensor_cores=False,
                supports_bf16=True,
                supports_cuda_graphs=False,
                tensor_core_arch="None (CPU Only)"
            )

    @classmethod
    def plan_execution(
        cls,
        model_size_mb: float,
        num_layers: int,
        config: Any,
        profile: Optional[HardwareProfile] = None,
        max_seq_len: int = 2048,
        batch_size: int = 1
    ) -> ExecutionPlan:
        """
        Execution Planner:
        Computes maximum GPU layer occupancy given VRAM headroom, KV cache budget, and layer footprint.
        """
        if profile is None:
            profile = cls.probe()

        if profile.device_type == "cpu" or profile.free_vram_mb <= cls.SAFETY_HEADROOM_MB:
            return ExecutionPlan(
                mode="CPU_RESIDENT",
                gpu_layers=0,
                cpu_layers=num_layers,
                enable_cuda_graphs=False,
                target_device="cpu",
                kv_cache_budget_mb=500.0,
                safety_margin_mb=0.0,
                chunk_size_prefill=128,
                recommended_batch_size=1
            )

        # 1. Estimate KV cache footprint per token per layer
        # Size per token = 2 (K, V) * num_key_value_heads * head_dim * bytes_per_elem
        num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 16))
        hidden_size = getattr(config, 'hidden_size', 2048)
        num_attn_heads = getattr(config, 'num_attention_heads', 16)
        head_dim = getattr(config, 'head_dim', hidden_size // num_attn_heads)

        # 2 bytes per element (FP16/BF16)
        kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * 2
        total_kv_mb = (kv_bytes_per_token_per_layer * max_seq_len * batch_size * num_layers) / (1024**2)

        # Available budget for weights
        usable_vram = profile.free_vram_mb - cls.SAFETY_HEADROOM_MB - total_kv_mb

        # Estimate bytes per layer
        # Model size includes embeddings + lm_head (~20%) and layers (~80%)
        layer_fraction = 0.82
        layer_weights_mb = (model_size_mb * layer_fraction) / max(1, num_layers)
        base_overhead_mb = model_size_mb * (1.0 - layer_fraction) # Embeddings + Norms

        available_for_layers = usable_vram - base_overhead_mb

        # 2. Decision Logic
        if model_size_mb + total_kv_mb <= (profile.free_vram_mb - cls.SAFETY_HEADROOM_MB):
            # Mode A: Full GPU Resident
            return ExecutionPlan(
                mode="FULL_GPU_RESIDENT",
                gpu_layers=num_layers,
                cpu_layers=0,
                enable_cuda_graphs=profile.supports_cuda_graphs,
                target_device=f"cuda:{profile.device_index}",
                kv_cache_budget_mb=round(total_kv_mb, 2),
                safety_margin_mb=cls.SAFETY_HEADROOM_MB,
                chunk_size_prefill=256,
                recommended_batch_size=min(4, max(1, int(usable_vram // (total_kv_mb or 1))))
            )
        else:
            # Mode B: Hybrid Dynamic Layer Offloading
            # Calculate how many layers can safely fit in VRAM, subtracting 2 slots for Ping-Pong DMA
            fit_layers = int(max(0, (available_for_layers // layer_weights_mb) - 2))
            fit_layers = min(num_layers - 1, max(0, fit_layers))

            return ExecutionPlan(
                mode="HYBRID_DYNAMIC_OFFLOAD",
                gpu_layers=fit_layers,
                cpu_layers=num_layers - fit_layers,
                enable_cuda_graphs=False, # Graphs require full static memory residency
                target_device=f"cuda:{profile.device_index}",
                kv_cache_budget_mb=round(total_kv_mb, 2),
                safety_margin_mb=cls.SAFETY_HEADROOM_MB,
                chunk_size_prefill=128,
                recommended_batch_size=1
            )
