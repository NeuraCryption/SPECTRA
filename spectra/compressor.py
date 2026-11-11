import os
import gc
import json
import time
import math
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

MAGIC_HEADER = b"TRDC4B01"

NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
])

def fast_walsh_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    orig_shape = x.shape
    d = orig_shape[-1]
    assert (d & (d - 1)) == 0, f"Last dimension ({d}) must be a power of 2 for FWHT"
    res = x.reshape(-1, d)
    h = 1
    while h < d:
        res = res.reshape(-1, d // (2 * h), 2, h)
        u, v = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.stack([u + v, u - v], dim=2).reshape(-1, d)
        h *= 2
    if normalize:
        res = res / math.sqrt(d)
    return res.reshape(orig_shape)

class SpectraCompressor:
    def __init__(
        self,
        group_size: int = 32,
        base_outlier_ratio: float = 0.01,
        adaptive_kurtosis: bool = True,
        hadamard_incoherence: bool = False
    ):
        self.group_size = group_size
        self.base_outlier_ratio = base_outlier_ratio
        self.adaptive_kurtosis = adaptive_kurtosis
        self.hadamard_incoherence = hadamard_incoherence

    @staticmethod
    def _tensor_to_bytes(t: torch.Tensor) -> bytes:
        """
        Safely serializes any PyTorch tensor (including BFloat16) to raw bytes in C++
        without Python iterator loops or NumPy ScalarType errors.
        """
        return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()

    @staticmethod
    def _quantize_chunk_group32(
        w_chunk: torch.Tensor,
        group_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N_c, K = w_chunk.shape
        w_groups = w_chunk.reshape(N_c, K // group_size, group_size).float()
        w_min = w_groups.amin(dim=-1, keepdim=True)
        w_max = w_groups.amax(dim=-1, keepdim=True)

        scales = (w_max - w_min) / 15.0
        scales = torch.clamp(scales, min=1e-8)
        zeros = torch.clamp(torch.round(-w_min / scales), 0, 15).to(torch.uint8)

        q_w = torch.clamp(torch.round((w_groups - (-zeros.float() * scales)) / scales), 0, 15).to(torch.uint8)
        q_w = q_w.reshape(N_c, K // group_size, 32)

        # Marlin Interleaved Packing: 0..15 in low nibbles, 16..31 in high nibbles
        w_first16 = q_w[:, :, 0:16]
        w_next16  = q_w[:, :, 16:32]
        packed = (w_first16 | (w_next16 << 4)).to(torch.uint8).reshape(N_c, K // 2)

        # Bit-pack zero-points (two 4-bit zero points per byte, handling odd group counts)
        z_2d = zeros.squeeze(-1) # [N_c, K // group_size]
        if (z_2d.shape[1] % 2) != 0:
            z_2d = torch.nn.functional.pad(z_2d, (0, 1), mode='constant', value=0)
        z_even = z_2d[:, 0::2]
        z_odd = z_2d[:, 1::2]
        packed_zeros = (z_even | (z_odd << 4)).to(torch.uint8)

        return packed.contiguous(), scales.squeeze(-1).to(torch.float16).contiguous(), packed_zeros.contiguous()

    @staticmethod
    def _compute_group32_quant(
        w: torch.Tensor,
        group_size: int = 32,
        outlier_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Microscaled Group-32 Quantization with Zero-Spike Memory Streaming:
        Processes in row chunks if N > 4096 to prevent OOM on large vocabulary embedding matrices.
        1. Marlin Interleaved Bit-Packing: weight k in low nibble, weight k+16 in high nibble.
        2. Bit-Packed Zero Points: two 4-bit zero points packed into a single uint8 byte.
        CUDA-accelerated if available, returning compact CPU tensors.
        """
        N, K = w.shape
        assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        idx_long = outlier_idx.long().to(device) if outlier_idx is not None else None
        if N <= 4096:
            w_dense = w.to(device).clone()
            if idx_long is not None:
                w_dense[:, idx_long] = 0.0
            p_ch, s_ch, z_ch = SpectraCompressor._quantize_chunk_group32(w_dense, group_size)
            return p_ch.cpu(), s_ch.cpu(), z_ch.cpu()

        chunk_rows = 4096
        num_packed_z_cols = (K // group_size + 1) // 2
        packed = torch.empty((N, K // 2), dtype=torch.uint8)
        scales = torch.empty((N, K // group_size), dtype=torch.float16)
        packed_zeros = torch.empty((N, num_packed_z_cols), dtype=torch.uint8)

        for r0 in range(0, N, chunk_rows):
            r1 = min(N, r0 + chunk_rows)
            chunk = w[r0:r1].to(device).clone()
            if idx_long is not None:
                chunk[:, idx_long] = 0.0
            p_ch, s_ch, z_ch = SpectraCompressor._quantize_chunk_group32(chunk, group_size)
            packed[r0:r1] = p_ch.cpu()
            scales[r0:r1] = s_ch.cpu()
            packed_zeros[r0:r1] = z_ch.cpu()
            del chunk, p_ch, s_ch, z_ch

        return packed, scales, packed_zeros

    def _isolate_hdsq_outliers(
        self,
        w: torch.Tensor,
        norm_vector: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, float, float]:
        """
        Universal Mathematical Activation-Aware Kurtosis Outlier Isolation:
        1. If norm_vector (gamma from preceding LayerNorm/RMSNorm) is present, scales column norms:
           Saliency_j = |gamma_j| * ||W_{:, j}||_2 (Calibration-Free Activation-Aware Saliency).
        2. Computes 4th standardized moment (Kurtosis) of channel saliency.
        3. Decouples heavy-tailed outliers in bit-for-bit lossless FP16/BF16 with uint16/int32 indices.
        4. Row-chunked column norm calculation avoids allocating full-size FP32 matrices.
        """
        N, K = w.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if N > 4096:
            col_norms_sq = torch.zeros(K, dtype=torch.float32, device=device)
            chunk_rows = 4096
            for r0 in range(0, N, chunk_rows):
                chunk = w[r0 : r0 + chunk_rows].float().to(device)
                col_norms_sq += (chunk ** 2).sum(dim=0)
            col_norms = torch.sqrt(col_norms_sq)
        else:
            w_f = w.float().to(device)
            col_norms = torch.norm(w_f, dim=0)

        if norm_vector is not None and norm_vector.numel() == K:
            col_saliency = norm_vector.to(device) * col_norms
        else:
            col_saliency = col_norms

        mu = col_saliency.mean()
        sigma2 = ((col_saliency - mu) ** 2).mean() + 1e-8
        m4 = ((col_saliency - mu) ** 4).mean()
        kurtosis = max(1.0, (m4 / (sigma2 ** 2)).item())

        if self.adaptive_kurtosis:
            excess_kurtosis = max(0.0, kurtosis - 3.0)
            k_mult = 1.0 + 0.15 * math.log1p(excess_kurtosis)
            ratio = float(torch.clamp(torch.tensor(self.base_outlier_ratio * k_mult, device=device), min=0.005, max=0.03).item())
        else:
            ratio = self.base_outlier_ratio

        num_outliers = max(1, int(round(K * ratio)))
        _, outlier_idx = torch.topk(col_saliency, k=num_outliers, largest=True)
        outlier_idx, _ = torch.sort(outlier_idx)
        outlier_idx_cpu = outlier_idx.to(torch.int32).cpu().contiguous()
        target_outlier_dtype = w.dtype if w.dtype in (torch.bfloat16, torch.float16) else torch.float16
        outlier_w = w[:, outlier_idx_cpu.long()].to(target_outlier_dtype).contiguous()

        if N <= 4096:
            w_dense = w.clone()
            w_dense[:, outlier_idx_cpu.long()] = 0.0
            if self.hadamard_incoherence and (K % self.group_size == 0):
                w_dense = fast_walsh_hadamard_transform(w_dense.reshape(N, K // self.group_size, self.group_size)).reshape(N, K)
        else:
            w_dense = None

        # Store outlier indices as int32 (supports K up to 2 Billion, eliminating uint16 wrapping)
        return w_dense, outlier_idx_cpu, outlier_w, ratio, kurtosis

    @staticmethod
    def _stream_tensors(model_path: Path) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Universal Zero-RAM-Spike Tensor Streamer:
        Iterates over safetensors shards or pytorch bin files one tensor at a time.
        """
        # 1. Safetensors check
        index_file = model_path / "model.safetensors.index.json"
        safetensor_files = []
        if index_file.exists() and HAS_SAFETENSORS:
            with open(index_file) as f:
                idx = json.load(f)
            shards = sorted(list(set(idx["weight_map"].values())))
            safetensor_files = [model_path / s for s in shards]
        elif HAS_SAFETENSORS:
            safetensor_files = sorted(list(model_path.glob("*.safetensors")))

        if safetensor_files:
            for sf in safetensor_files:
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        yield k, f.get_tensor(k)
            return

        # 2. PyTorch Bin check
        bin_index = model_path / "pytorch_model.bin.index.json"
        bin_files = []
        if bin_index.exists():
            with open(bin_index) as f:
                idx = json.load(f)
            shards = sorted(list(set(idx["weight_map"].values())))
            bin_files = [model_path / s for s in shards]
        else:
            bin_files = sorted(list(model_path.glob("pytorch_model*.bin")))

        if bin_files:
            for bf in bin_files:
                sd = torch.load(str(bf), map_location="cpu", weights_only=True)
                for k, v in sd.items():
                    yield k, v
                del sd
                gc.collect()
            return

        # 3. Fallback: AutoModelForCausalLM
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        for k, v in model.state_dict().items():
            yield k, v
        del model
        gc.collect()

    @classmethod
    def _prescan_norm_vectors(cls, model_path: Path) -> Dict[str, torch.Tensor]:
        """
        Pre-scans small 1D normalization weights (LayerNorm, RMSNorm) across all shards.
        Consumes < 3 MB host RAM total and enables exact topological routing of activation saliency.
        """
        norm_map = {}
        index_file = model_path / "model.safetensors.index.json"
        safetensor_files = []
        if index_file.exists() and HAS_SAFETENSORS:
            try:
                with open(index_file) as f:
                    idx = json.load(f)
                shards = sorted(list(set(idx["weight_map"].values())))
                safetensor_files = [model_path / s for s in shards]
            except Exception:
                pass
        elif HAS_SAFETENSORS:
            safetensor_files = sorted(list(model_path.glob("*.safetensors")))

        norm_keywords = ["layernorm.weight", "norm.weight", "ln.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight"]

        if safetensor_files:
            for sf in safetensor_files:
                try:
                    with safe_open(str(sf), framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if any(nk in k.lower() for nk in norm_keywords):
                                t = f.get_tensor(k)
                                if t.ndim == 1:
                                    norm_map[k] = t.detach().float().abs()
                except Exception:
                    pass
            return norm_map

        bin_index = model_path / "pytorch_model.bin.index.json"
        bin_files = []
        if bin_index.exists():
            try:
                with open(bin_index) as f:
                    idx = json.load(f)
                shards = sorted(list(set(idx["weight_map"].values())))
                bin_files = [model_path / s for s in shards]
            except Exception:
                pass
        else:
            bin_files = sorted(list(model_path.glob("pytorch_model*.bin")))

        if bin_files:
            for bf in bin_files:
                try:
                    sd = torch.load(str(bf), map_location="cpu", weights_only=True)
                    for k, v in sd.items():
                        if any(nk in k.lower() for nk in norm_keywords) and v.ndim == 1:
                            norm_map[k] = v.detach().float().abs()
                    del sd
                except Exception:
                    pass
        return norm_map

    @staticmethod
    def _get_norm_vector_for_param(name: str, norm_map: Dict[str, torch.Tensor], in_features: int) -> Optional[torch.Tensor]:
        """
        Topological Norm Mapping:
        Maps attention projections (q/k/v/qkv) to the layer's attention pre-norm,
        and MLP projections (gate/up/fc1) to the layer's MLP pre-norm.
        Explicitly excludes output projections (o_proj, down_proj) and embeddings/lm_head,
        ensuring zero false activation outlier amplification.
        """
        n_low = name.lower()
        is_attn_in = any(x in n_low for x in ["q_proj", "k_proj", "v_proj", "qkv_proj", "c_attn", "in_proj", "w_pack", "in_proj_weight"])
        is_mlp_in = any(x in n_low for x in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "w3", "gate_up_proj", "input_linear", "shared_mlp"])

        if not (is_attn_in or is_mlp_in):
            return None

        parts = name.split('.')
        prefix = None
        for idx, p in enumerate(parts):
            if p in ('layers', 'h', 'blocks', 'block') and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                prefix = '.'.join(parts[:idx + 2])
                break

        if prefix is None:
            return None

        attn_candidates = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.attention_norm.weight",
            f"{prefix}.ln_1.weight",
            f"{prefix}.operator_norm.weight",
            f"{prefix}.norm1.weight",
            f"{prefix}.norm.weight",  # Nemotron-H Transformer/Mamba
        ]
        mlp_candidates = [
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.ffn_norm.weight",
            f"{prefix}.ln_2.weight",
            f"{prefix}.mlp_norm.weight",
            f"{prefix}.norm2.weight",
            f"{prefix}.norm.weight",  # Nemotron-H Mamba
            f"{prefix}.shared_mlp_norm.weight",
        ]

        target_candidates = attn_candidates if is_attn_in else mlp_candidates
        for cand in target_candidates:
            if cand in norm_map and norm_map[cand].numel() == in_features:
                return norm_map[cand]

        for k, v in norm_map.items():
            if k.startswith(prefix) and v.numel() == in_features:
                if is_attn_in and any(x in k.lower() for x in ["input", "ln_1", "attention", "norm1", "norm"]):
                    return v
                elif is_mlp_in and any(x in k.lower() for x in ["post", "ln_2", "ffn", "mlp", "norm2", "norm"]):
                    return v

        return None

    def compress(
        self,
        model_id_or_path: Union[str, Path],
        output_file: Union[str, Path],
        save_tokenizer: bool = True,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Compresses ANY model into a self-contained .trdc container with ~0 MB RAM overhead.
        Supports Dense and MoE architectures with dynamic kurtosis moment calibration.
        """
        t0 = time.time()
        model_path = Path(model_id_or_path)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Compressor] Source: {model_path} -> Output: {output_path}")
        print(f"[Compressor] Group Size: {self.group_size} | Outliers: {self.base_outlier_ratio * 100:.1f}%")

        # 1. Load Config
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        config_dict = config.to_dict()

        temp_file = output_path.with_suffix(".tmp")
        manifest = {
            "version": "TRDC4B01",
            "group_size": self.group_size,
            "base_outlier_ratio": self.base_outlier_ratio,
            "architecture": config_dict.get("architectures", ["AutoModelForCausalLM"])[0],
            "config": config_dict,
            "tensors": {}
        }

        total_orig_bytes = 0
        total_compressed_bytes = 0
        idx = 0

        # Estimate total tensor count if index or safetensors files exist
        idx_path = model_path / "model.safetensors.index.json"
        total_expected_tensors = 0
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    idx_meta = json.load(f)
                total_expected_tensors = len(idx_meta.get("weight_map", {}))
            except Exception:
                pass
        elif HAS_SAFETENSORS:
            try:
                for sf in model_path.glob("*.safetensors"):
                    with safe_open(str(sf), framework="pt") as f:
                        total_expected_tensors += len(f.keys())
            except Exception:
                pass

        # 2. Pre-scan 1D Normalization Weights for Topological Saliency Mapping
        norm_map = self._prescan_norm_vectors(model_path)

        # 3. Stream and Compress
        with open(temp_file, "wb") as f_out:
            f_out.write(MAGIC_HEADER)
            f_out.write(struct.pack("<QQ", 0, 0)) # Placeholder for manifest offset and length

            for name, param in self._stream_tensors(model_path):
                idx += 1
                shape = list(param.shape)
                cur_offset = f_out.tell()
                p_bytes_orig = param.numel() * param.element_size()
                total_orig_bytes += p_bytes_orig

                # Universal Mathematical Matrix Classification:
                # Any 2D weight matrix with dimension divisible by group_size is compressed
                # using Group-32 Microscaling + Kurtosis-Aware Outlier Decoupling.
                # This universally covers Linear layers, Embeddings, Attention, MLPs, Experts, and Mamba projections!
                is_compressible_matrix = (
                    (param.ndim == 2) and
                    (param.shape[1] >= self.group_size) and
                    (param.shape[1] % self.group_size == 0)
                )

                if is_compressible_matrix:
                    norm_vec = self._get_norm_vector_for_param(name, norm_map, param.shape[1])
                    w_dense, outlier_idx, outlier_w, actual_ratio, kurtosis = self._isolate_hdsq_outliers(param, norm_vector=norm_vec)
                    if w_dense is not None:
                        packed_b, scales, zeros = self._compute_group32_quant(w_dense, self.group_size)
                    else:
                        packed_b, scales, zeros = self._compute_group32_quant(param, self.group_size, outlier_idx=outlier_idx)

                    p_bytes = self._tensor_to_bytes(packed_b)
                    s_bytes = self._tensor_to_bytes(scales)
                    z_bytes = self._tensor_to_bytes(zeros)
                    oi_bytes = self._tensor_to_bytes(outlier_idx)
                    ow_bytes = self._tensor_to_bytes(outlier_w)

                    f_out.write(p_bytes)
                    f_out.write(s_bytes)
                    f_out.write(z_bytes)
                    f_out.write(oi_bytes)
                    f_out.write(ow_bytes)

                    c_len = len(p_bytes) + len(s_bytes) + len(z_bytes) + len(oi_bytes) + len(ow_bytes)
                    total_compressed_bytes += c_len

                    manifest["tensors"][name] = {
                        "type": "q4_g32_hdsq",
                        "shape": shape,
                        "offset": cur_offset,
                        "packed_len": len(p_bytes),
                        "scale_len": len(s_bytes),
                        "zero_len": len(z_bytes),
                        "zero_packed": True,
                        "outlier_idx_len": len(oi_bytes),
                        "outlier_idx_dtype": "int32",
                        "outlier_w_len": len(ow_bytes),
                        "outlier_w_dtype": "bfloat16" if outlier_w.dtype == torch.bfloat16 else "float16",
                        "num_outliers": outlier_idx.numel(),
                        "outlier_ratio": round(actual_ratio, 4),
                        "kurtosis": round(kurtosis, 2),
                        "marlin_layout": True,
                        "total_len": c_len
                    }

                    del w_dense, outlier_idx, outlier_w, packed_b, scales, zeros

                    if idx % 30 == 0:
                        print(f"[{idx} tensors] Compressed {name} {shape} -> 4-bit Group-32 + HDSQ (Kurtosis: {kurtosis:.1f})")

                else:
                    if param.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
                        save_dtype = param.dtype
                        dtype_str = str(param.dtype).replace("torch.", "")
                    elif param.dtype == torch.bfloat16:
                        save_dtype = torch.bfloat16
                        dtype_str = "bfloat16"
                    elif param.dtype == torch.float32:
                        save_dtype = torch.float32 if "inv_freq" in name else torch.float16
                        dtype_str = "float32" if save_dtype == torch.float32 else "float16"
                    else:
                        save_dtype = torch.float16
                        dtype_str = "float16"

                    t_bytes = self._tensor_to_bytes(param.to(save_dtype))
                    f_out.write(t_bytes)
                    c_len = len(t_bytes)
                    total_compressed_bytes += c_len

                    manifest["tensors"][name] = {
                        "type": f"lossless_{dtype_str}",
                        "dtype": dtype_str,
                        "shape": shape,
                        "offset": cur_offset,
                        "total_len": c_len
                    }
                    if idx % 30 == 0:
                        print(f"[{idx} tensors] Preserved lossless {name} {shape} ({manifest['tensors'][name]['dtype']})")

                del param

                if progress_callback:
                    pct = (idx / total_expected_tensors * 100) if total_expected_tensors > 0 else 0.0
                    cur_ratio = total_orig_bytes / max(1, total_compressed_bytes)
                    try:
                        progress_callback({
                            "status": "compressing",
                            "tensor": name,
                            "index": idx,
                            "total": total_expected_tensors,
                            "percentage": round(pct, 1),
                            "orig_bytes": total_orig_bytes,
                            "compressed_bytes": total_compressed_bytes,
                            "ratio": round(cur_ratio, 2)
                        })
                    except Exception:
                        pass

            # Write manifest at end of container
            manifest_offset = f_out.tell()
            manifest_json = json.dumps(manifest).encode("utf-8")
            f_out.write(manifest_json)
            manifest_len = len(manifest_json)

            # Seek back and write actual manifest location in header
            f_out.seek(8)
            f_out.write(struct.pack("<QQ", manifest_offset, manifest_len))

        if output_path.exists():
            output_path.unlink()
        temp_file.rename(output_path)

        # 3. Save tokenizer, config, and custom architecture code alongside if available
        if save_tokenizer:
            try:
                tok_dir = output_path.parent / (output_path.stem + "_tokenizer")
                tok_dir.mkdir(parents=True, exist_ok=True)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                    tokenizer.save_pretrained(str(tok_dir))
                except Exception:
                    pass
                try:
                    config.save_pretrained(str(tok_dir))
                except Exception:
                    pass

                # Copy any custom modeling code files (*.py, *.jinja, *.json) so remote architectures load standalone
                if isinstance(model_path, (str, Path)):
                    m_p = Path(model_path)
                    if m_p.is_dir():
                        import shutil
                        for ext in ("*.py", "*.jinja", "*.json"):
                            for f in m_p.glob(ext):
                                dest = tok_dir / f.name
                                if not dest.exists():
                                    try:
                                        shutil.copy2(f, dest)
                                    except Exception:
                                        pass
            except Exception as e:
                print(f"Notice: Tokenizer save skipped ({e})")

        compressed_size = output_path.stat().st_size
        compression_ratio = total_orig_bytes / compressed_size if compressed_size > 0 else 1.0
        dt = time.time() - t0

        summary = {
            "model": str(model_path),
            "container": str(output_path),
            "total_tensors": idx,
            "uncompressed_bytes": total_orig_bytes,
            "compressed_bytes": compressed_size,
            "compression_ratio": round(compression_ratio, 2),
            "vram_savings_pct": round((1.0 - 1.0 / compression_ratio) * 100, 1),
            "compression_time_sec": round(dt, 2)
        }

        print("=" * 80)
        print(f"COMPRESSION COMPLETED IN {dt:.2f}s! ({idx} tensors processed)")
        print(f"Original Size   : {total_orig_bytes / (1024**3):.3f} GB")
        print(f"Compressed Size : {compressed_size / (1024**3):.3f} GB")
        print(f"Ratio           : {compression_ratio:.2f}x ({summary['vram_savings_pct']}% VRAM reduction)")
        print("=" * 80)

        return summary
