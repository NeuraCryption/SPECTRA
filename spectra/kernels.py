import math
import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['N', 'K']
)
@triton.jit
def _fused_w4a16_gemv_hdsq_kernel(
    x_ptr,            # [1, K]
    packed_w_ptr,     # [N, K // 2]
    scales_ptr,       # [N, K // 32]
    zeros_ptr,        # [N, K // 32]
    out_ptr,          # [1, N]
    bias_ptr,         # [N] or None
    outlier_idx_ptr,  # [num_outliers] or None
    outlier_w_ptr,    # [N, num_outliers] or None
    stride_xb, stride_xk,
    stride_wn, stride_wk_packed,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_ob, stride_on,
    stride_own, stride_owk,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_OUTLIERS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_OUTLIERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MARLIN_LAYOUT: tl.constexpr = False,
    IS_BF16: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # 1. Main 4-bit Group-32 Microscaled Reduction Loop
    for k_start in range(0, K, BLOCK_K):
        offs_k_packed = (k_start // 2) + tl.arange(0, BLOCK_K // 2)
        ptr_w = packed_w_ptr + (offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk_packed)
        mask_w = mask_n[:, None] & (offs_k_packed[None, :] < (K // 2))
        packed_bytes = tl.load(ptr_w, mask=mask_w, other=0)

        # Unpack low and high nibbles
        low_nibble = (packed_bytes & 0x0F).to(tl.float32)
        high_nibble = ((packed_bytes >> 4) & 0x0F).to(tl.float32)

        # Load scales and zeros for Group-32
        offs_k_group = (k_start // 32) + tl.arange(0, BLOCK_K // 32)
        ptr_s = scales_ptr + (offs_n[:, None] * stride_sn + offs_k_group[None, :] * stride_sk)
        ptr_z = zeros_ptr + (offs_n[:, None] * stride_zn + offs_k_group[None, :] * stride_zk)
        mask_sz = mask_n[:, None] & (offs_k_group[None, :] < (K // 32))

        s_val = tl.load(ptr_s, mask=mask_sz, other=1.0).to(tl.float32)
        z_val = tl.load(ptr_z, mask=mask_sz, other=0.0).to(tl.float32)

        # Repeat scale & zero across the 16 bytes of each 32-weight group
        s_expanded = tl.reshape(s_val[:, :, None] + tl.zeros((1, 1, 16), dtype=tl.float32), (BLOCK_N, BLOCK_K // 2))
        z_expanded = tl.reshape(z_val[:, :, None] + tl.zeros((1, 1, 16), dtype=tl.float32), (BLOCK_N, BLOCK_K // 2))

        # In-register dequantization
        w_low = (low_nibble - z_expanded) * s_expanded
        w_high = (high_nibble - z_expanded) * s_expanded

        # Load activation vector elements
        offs_k_byte = tl.arange(0, BLOCK_K // 2)
        if MARLIN_LAYOUT:
            # Marlin layout: 0..15 in low nibbles, 16..31 in high nibbles per 16-byte block
            g_idx = offs_k_byte // 16
            rem_idx = offs_k_byte % 16
            k_low = k_start + g_idx * 32 + rem_idx
            k_high = k_low + 16

            x_low = tl.load(x_ptr + k_low * stride_xk, mask=k_low < K, other=0.0).to(tl.float32)
            x_high = tl.load(x_ptr + k_high * stride_xk, mask=k_high < K, other=0.0).to(tl.float32)

            acc += tl.sum(w_low * x_low[None, :], axis=1)
            acc += tl.sum(w_high * x_high[None, :], axis=1)
        else:
            # Even-Odd layout: 2k in low nibble, 2k+1 in high nibble
            offs_even = k_start + offs_k_byte * 2
            offs_odd = offs_even + 1
            x_even = tl.load(x_ptr + offs_even * stride_xk, mask=offs_even < K, other=0.0).to(tl.float32)
            x_odd  = tl.load(x_ptr + offs_odd * stride_xk,  mask=offs_odd < K,  other=0.0).to(tl.float32)

            acc += tl.sum(w_low * x_even[None, :], axis=1)
            acc += tl.sum(w_high * x_odd[None, :], axis=1)


    # 2. IN-KERNEL LOSSLESS FP16 OUTLIER ACCUMULATION (HDSQ-Ultra)
    if HAS_OUTLIERS:
        BLOCK_OUT: tl.constexpr = 32
        for out_start in range(0, NUM_OUTLIERS, BLOCK_OUT):
            offs_out = out_start + tl.arange(0, BLOCK_OUT)
            mask_out = offs_out < NUM_OUTLIERS
            
            # Load outlier column indices
            col_idx = tl.load(outlier_idx_ptr + offs_out, mask=mask_out, other=0)
            
            # Gather activation values at outlier indices
            x_out_val = tl.load(x_ptr + col_idx * stride_xk, mask=mask_out, other=0.0).to(tl.float32)
            
            # Load exact FP16 outlier weights
            ptr_ow = outlier_w_ptr + (offs_n[:, None] * stride_own + offs_out[None, :] * stride_owk)
            mask_ow = mask_n[:, None] & mask_out[None, :]
            ow_val = tl.load(ptr_ow, mask=mask_ow, other=0.0).to(tl.float32)
            
            # Fused dot-product accumulation
            acc += tl.sum(ow_val * x_out_val[None, :], axis=1)

    if HAS_BIAS:
        b_val = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += b_val

    if IS_BF16:
        tl.store(out_ptr + offs_n * stride_on, acc.to(tl.bfloat16), mask=mask_n)
    else:
        tl.store(out_ptr + offs_n * stride_on, acc.to(tl.float16), mask=mask_n)


def fused_w4a16_gemv_hdsq(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    outlier_idx: Optional[torch.Tensor] = None,
    outlier_w: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    marlin_layout: bool = False
) -> torch.Tensor:
    orig_dim = x.ndim
    if orig_dim == 1:
        x = x.unsqueeze(0)
    B, K = x.shape
    N = packed_w.shape[0]

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)

    has_bias = bias is not None
    has_outliers = (outlier_idx is not None) and (outlier_w is not None)
    num_outliers = outlier_idx.numel() if has_outliers else 0

    dummy_idx = torch.empty(0, dtype=torch.int64, device=x.device)
    dummy_w = torch.empty((0, 0), dtype=torch.float16, device=x.device)
    dummy_b = torch.empty(0, dtype=x.dtype, device=x.device)

    _fused_w4a16_gemv_hdsq_kernel[grid](
        x, packed_w, scales, zeros, out,
        bias if has_bias else dummy_b,
        outlier_idx if has_outliers else dummy_idx,
        outlier_w if has_outliers else dummy_w,
        x.stride(0), x.stride(1),
        packed_w.stride(0), packed_w.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        out.stride(0), out.stride(1),
        outlier_w.stride(0) if has_outliers else 0,
        outlier_w.stride(1) if has_outliers else 0,
        N=N, K=K,
        NUM_OUTLIERS=num_outliers,
        HAS_BIAS=has_bias,
        HAS_OUTLIERS=has_outliers,
        MARLIN_LAYOUT=marlin_layout,
        IS_BF16=(x.dtype == torch.bfloat16),
    )

    if orig_dim == 1:
        out = out.squeeze(0)
    return out


# =============================================================================
# KERNEL 2: BATCHED W4A16 TENSOR CORE GEMM (B >= 2 or S > 1)
# =============================================================================

@triton.jit
def _batched_w4a16_gemm_kernel(
    a_ptr,            # [M, K]
    packed_b_ptr,     # [N, K // 2]
    scales_ptr,       # [N, K // 32]
    zeros_ptr,        # [N, K // 32]
    c_ptr,            # [M, N]
    outlier_idx_ptr,  # [num_outliers]
    outlier_w_ptr,    # [N, num_outliers]
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_cm, stride_cn,
    stride_own, stride_owk,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_OUTLIERS: tl.constexpr,
    HAS_OUTLIERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MARLIN_LAYOUT: tl.constexpr = False,
    IS_BF16: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_byte = tl.arange(0, BLOCK_K // 2)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k_packed = (k_start // 2) + offs_k_byte
        ptr_b = packed_b_ptr + (offs_bn[None, :] * stride_bn + offs_k_packed[:, None] * stride_bk)
        mask_b = (offs_bn[None, :] < N) & (offs_k_packed[:, None] < (K // 2))
        packed_bytes = tl.load(ptr_b, mask=mask_b, other=0)

        low = (packed_bytes & 0x0F).to(tl.float32)
        high = ((packed_bytes >> 4) & 0x0F).to(tl.float32)

        offs_k_group = (k_start // 32) + tl.arange(0, BLOCK_K // 32)
        ptr_s = scales_ptr + (offs_bn[:, None] * stride_sn + offs_k_group[None, :] * stride_sk)
        ptr_z = zeros_ptr + (offs_bn[:, None] * stride_zn + offs_k_group[None, :] * stride_zk)
        mask_sz = (offs_bn[:, None] < N) & (offs_k_group[None, :] < (K // 32))

        s_tile = tl.load(ptr_s, mask=mask_sz, other=1.0).to(tl.float32)
        z_tile = tl.load(ptr_z, mask=mask_sz, other=0.0).to(tl.float32)

        s_exp_n = tl.reshape(s_tile[:, :, None] + tl.zeros((1, 1, 16), dtype=tl.float32), (BLOCK_N, BLOCK_K // 2))
        z_exp_n = tl.reshape(z_tile[:, :, None] + tl.zeros((1, 1, 16), dtype=tl.float32), (BLOCK_N, BLOCK_K // 2))
        s_exp = tl.trans(s_exp_n)
        z_exp = tl.trans(z_exp_n)

        if IS_BF16:
            w_low = ((low - z_exp) * s_exp).to(tl.bfloat16)
            w_high = ((high - z_exp) * s_exp).to(tl.bfloat16)
        else:
            w_low = ((low - z_exp) * s_exp).to(tl.float16)
            w_high = ((high - z_exp) * s_exp).to(tl.float16)

        if MARLIN_LAYOUT:
            g_idx = offs_k_byte // 16
            rem_idx = offs_k_byte % 16
            k_low = k_start + g_idx * 32 + rem_idx
            k_high = k_low + 16

            ptr_a_low = a_ptr + (offs_am[:, None] * stride_am + k_low[None, :] * stride_ak)
            ptr_a_high = a_ptr + (offs_am[:, None] * stride_am + k_high[None, :] * stride_ak)
            mask_a_low = (offs_am[:, None] < M) & (k_low[None, :] < K)
            mask_a_high = (offs_am[:, None] < M) & (k_high[None, :] < K)

            if IS_BF16:
                a_low = tl.load(ptr_a_low, mask=mask_a_low, other=0.0).to(tl.bfloat16)
                a_high = tl.load(ptr_a_high, mask=mask_a_high, other=0.0).to(tl.bfloat16)
            else:
                a_low = tl.load(ptr_a_low, mask=mask_a_low, other=0.0).to(tl.float16)
                a_high = tl.load(ptr_a_high, mask=mask_a_high, other=0.0).to(tl.float16)

            acc += tl.dot(a_low, w_low)
            acc += tl.dot(a_high, w_high)
        else:
            k_even = k_start + offs_k_byte * 2
            k_odd  = k_even + 1

            ptr_a_even = a_ptr + (offs_am[:, None] * stride_am + k_even[None, :] * stride_ak)
            ptr_a_odd  = a_ptr + (offs_am[:, None] * stride_am + k_odd[None, :] * stride_ak)
            mask_a_even = (offs_am[:, None] < M) & (k_even[None, :] < K)
            mask_a_odd  = (offs_am[:, None] < M) & (k_odd[None, :] < K)

            if IS_BF16:
                a_even = tl.load(ptr_a_even, mask=mask_a_even, other=0.0).to(tl.bfloat16)
                a_odd  = tl.load(ptr_a_odd,  mask=mask_a_odd,  other=0.0).to(tl.bfloat16)
            else:
                a_even = tl.load(ptr_a_even, mask=mask_a_even, other=0.0).to(tl.float16)
                a_odd  = tl.load(ptr_a_odd,  mask=mask_a_odd,  other=0.0).to(tl.float16)

            acc += tl.dot(a_even, w_low)
            acc += tl.dot(a_odd, w_high)

    if HAS_OUTLIERS:
        BLOCK_OUT: tl.constexpr = 32
        for out_start in range(0, NUM_OUTLIERS, BLOCK_OUT):
            offs_out = out_start + tl.arange(0, BLOCK_OUT)
            mask_out = offs_out < NUM_OUTLIERS
            col_idx = tl.load(outlier_idx_ptr + offs_out, mask=mask_out, other=0)

            ptr_a_out = a_ptr + (offs_am[:, None] * stride_am + col_idx[None, :] * stride_ak)
            mask_a_out = (offs_am[:, None] < M) & mask_out[None, :]

            ptr_ow = outlier_w_ptr + (offs_bn[None, :] * stride_own + offs_out[:, None] * stride_owk)
            mask_ow = (offs_bn[None, :] < N) & mask_out[:, None]

            if IS_BF16:
                a_out_tile = tl.load(ptr_a_out, mask=mask_a_out, other=0.0).to(tl.bfloat16)
                ow_tile = tl.load(ptr_ow, mask=mask_ow, other=0.0).to(tl.bfloat16)
            else:
                a_out_tile = tl.load(ptr_a_out, mask=mask_a_out, other=0.0).to(tl.float16)
                ow_tile = tl.load(ptr_ow, mask=mask_ow, other=0.0).to(tl.float16)

            acc += tl.dot(a_out_tile, ow_tile)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ptr_c = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    if IS_BF16:
        tl.store(ptr_c, acc.to(tl.bfloat16), mask=mask_c)
    else:
        tl.store(ptr_c, acc.to(tl.float16), mask=mask_c)

def batched_w4a16_gemm(
    a: torch.Tensor,
    packed_b: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    outlier_idx: Optional[torch.Tensor] = None,
    outlier_w: Optional[torch.Tensor] = None,
    marlin_layout: bool = False
) -> torch.Tensor:
    orig_shape = a.shape
    M = a.numel() // a.shape[-1]
    K = a.shape[-1]
    N = packed_b.shape[0]

    is_bf16 = (a.dtype == torch.bfloat16)
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16

    a_2d = a.reshape(M, K)
    c = torch.empty((M, N), device=a.device, dtype=compute_dtype)

    has_outliers = (outlier_idx is not None) and (outlier_w is not None)
    num_outliers = outlier_idx.numel() if has_outliers else 0

    dummy_idx = torch.empty(0, dtype=torch.int64, device=a.device)
    dummy_w = torch.empty((0, 0), dtype=compute_dtype, device=a.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _batched_w4a16_gemm_kernel[grid](
        a_2d, packed_b, scales, zeros, c,
        outlier_idx if has_outliers else dummy_idx,
        outlier_w if has_outliers else dummy_w,
        a_2d.stride(0), a_2d.stride(1),
        packed_b.stride(0), packed_b.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        c.stride(0), c.stride(1),
        outlier_w.stride(0) if has_outliers else 0,
        outlier_w.stride(1) if has_outliers else 0,
        M=M, N=N, K=K,
        NUM_OUTLIERS=num_outliers,
        HAS_OUTLIERS=has_outliers,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        MARLIN_LAYOUT=marlin_layout,
        IS_BF16=is_bf16,
        num_warps=4,
        num_stages=2
    )
    return c.reshape(orig_shape[:-1] + (N,)).to(a.dtype)
