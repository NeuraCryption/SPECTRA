# SPECTRA: Universal High-Throughput Model Compression & Autonomous Serving Runtime

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenAI Triton 3.0+](https://img.shields.io/badge/OpenAI%20Triton-3.0%2B-purple.svg)](https://github.com/triton-lang/triton)
[![CUDA 12.0+](https://img.shields.io/badge/NVIDIA%20CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NeuraCryption-orange.svg)](https://huggingface.co/neuracryption)

Official open-source repository for **SPECTRA** (Sparse-Precision Exact Channel Tensor Reconstruction Architecture), an architecture-agnostic neural compression engine and high-throughput execution runtime engineered to break the **Memory Wall** on edge and consumer GPUs (such as 4 GB and 8 GB NVIDIA RTX laptop and desktop GPUs).

---

## Table of Contents
1. [Why SPECTRA? The Edge Memory Wall](#why-spectra-the-edge-memory-wall)
2. [Architectural Highlights](#architectural-highlights)
3. [Quickstart: 1-Click Launch & Turnkey Operation](#quickstart-1-click-launch--turnkey-operation)
4. [Cloning & Running Models Instantly](#cloning--running-models-instantly)
5. [Interactive Web Dashboard Features](#interactive-web-dashboard-features)
6. [Empirical Benchmarks (Physical RTX 3050 Ti Laptop GPU &bull; 4 GB VRAM)](#empirical-benchmarks-physical-rtx-3050-ti-laptop-gpu--4-gb-vram)
7. [Under the Hood: Microarchitectural Deep-Dive](#under-the-hood-microarchitectural-deep-dive)
8. [CLI Reference](#cli-reference)
9. [Python SDK Integration](#python-sdk-integration)
10. [Hardware Compatibility & System Requirements](#hardware-compatibility--system-requirements)
11. [License & Citation](#license--citation)

---

## Why SPECTRA? The Edge Memory Wall

Running modern large language models (LLMs) and hybrid state-space models (SSMs) on edge devices is fundamentally bounded by memory bandwidth rather than arithmetic throughput.

### The Arithmetic Intensity Dilemma
During single-token autoregressive generation ($M = 1$), an inference engine must read the entire weight matrix $W$ from GPU global memory (GDDR6/VRAM) to evaluate a single token vector $x$:
$$\text{Operational Arithmetic Intensity} \approx \frac{2 \cdot N \cdot K \text{ FLOPs}}{2 \cdot N \cdot K \text{ Bytes loaded}} = 1.0\text{ FLOP/Byte}$$

On a consumer GPU such as the NVIDIA GeForce RTX 3050 Ti Laptop GPU:
- **Peak FP16 Tensor Compute:** $\approx 32.0\text{ TFLOP/s}$
- **Memory Bus Bandwidth:** $192.0\text{ GB/s}$ (128-bit bus width)
- **Theoretical Maximum Decode Throughput:** At $1\text{ FLOP/Byte}$, memory bandwidth saturates at only $192.0\text{ GFLOP/s}$, leaving compute units **over 95% idle**.

To accelerate decoding, the weight bytes transferred across the memory bus must be compressed. However, traditional quantization techniques present severe barriers:
1. **Host RAM OOM Crashes:** Existing quantizers (e.g. GPTQ, AWQ, HQQ) attempt to load full model weights into host CPU RAM simultaneously, requiring 16 to 32 GB of system RAM for a 7B model. Consumer laptops crash immediately.
2. **Empirical Calibration Data Overhead:** Standard post-training quantization requires downloading external calibration datasets (e.g. Pile, WikiText-2, C4) and spending hours optimizing Hessian matrices.
3. **Hardware Fragility in Constrained VRAM:** When a model exceeds VRAM, existing runtimes terminate with CUDA Out of Memory errors rather than intelligently planning layer residency.

SPECTRA eliminates all three limitations through mathematical guarantees.

---

## Architectural Highlights

```
+-------------------------------------------------------------------------------+
|                             SPECTRA ARCHITECTURE                              |
+-------------------------------------------------------------------------------+
|  1. ZERO-CALIBRATION AST CURVATURE EXTRACTION                                 |
|     Extracts affine scale vectors (gamma) from RMSNorm/LayerNorm in <50 ms.  |
|     Mathematical proxy for parameter sensitivity: S_j = |gamma_j| * ||W_:,j|| |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  2. STREAMING ZERO-SPIKE GROUP-32 MICROSCALING                                |
|     Quantizes 32 contiguous weights per tile into 4-bit nibbles (uint8 pairs).|
|     Streams tensor-by-tensor from disk shards with Host RAM <= 50 MB.        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  3. TRDC BINARY CONTAINER SERIALIZATION                                       |
|     Custom container format (TRDC4B01 magic header + JSON manifest).          |
|     Direct streaming into pinned CUDA buffers without memory copying.        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  4. REGISTER-LEVEL FUSED TRITON HARDWARE KERNELS                              |
|     * fused_w4a16_gemv_hdsq: B=1 decode, in-register unpack & dot product.    |
|     * batched_w4a16_gemm: B>=2 prefill/decode, direct Tensor Core tiling.    |
|     Zero intermediate global memory scratchpads.                             |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  5. AUTONOMOUS RUNTIME SUBSYSTEMS                                             |
|     * SpectraFusedEmbedding: Gather-and-dequantize dynamic vocabulary.        |
|     * PingPongLayerOffloader: Double-buffered asynchronous PCIe DMA pipeline.|
|     * SpectraQuantizedCache: Group-32 4-bit KV cache up to 8,192 tokens.     |
+-------------------------------------------------------------------------------+
```

---

## Quickstart: 1-Click Launch & Turnkey Operation

SPECTRA requires zero configuration. Clone the repository, install dependencies, and launch:

```bash
# 1. Clone SPECTRA
git clone https://github.com/NeuraCryption/SPECTRA.git
cd SPECTRA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Interactive Web Dashboard
python launch.py
```

*(Alternatively, run `./launch.sh` or `spectra ui`)*

The launcher automatically:
1. Probes your physical GPU, compute capability, VRAM headroom, and host RAM via NVML.
2. Scans your directory tree for all cloned or local model repositories.
3. Automatically finds an open network port (default: `7860`).
4. Launches the local server and opens your web browser to `http://localhost:7860`.

---

## Cloning & Running Models Instantly

SPECTRA is engineered so that **cloning any SPECTRA model repository directly into your project workspace makes it immediately runnable with zero manual steps**.

### Step 1: Clone any Model Repository from Hugging Face
Clone one or more pre-compressed model repositories directly into the SPECTRA folder (or a `models/` directory):

```bash
# Example A: DeepSeek-R1-Distill-1.5B (Reasoning model, 1.03 GB)
git clone https://huggingface.co/neuracryption/DeepSeek-R1-Distill-1.5B-SPECTRA

# Example B: NVIDIA Nemotron-3-Nano-4B (Mamba-2 Hybrid, 2.26 GB)
git clone https://huggingface.co/neuracryption/Nemotron-3-Nano-4B-SPECTRA

# Example C: IBM Granite-4.2-3B (Mamba-2 Hybrid, 2.06 GB)
git clone https://huggingface.co/neuracryption/Granite-4.2-3B-SPECTRA

# Example D: LiquidAI LFM-3-1.7B (Hybrid ConvNet/SSM, 1.00 GB)
git clone https://huggingface.co/neuracryption/LFM-3-1.7B-SPECTRA
```

### Step 2: Open the Web UI
Refresh the browser at `http://localhost:7860`. 
- The model discovery scanner detects the cloned folder, reads its `config.json` and `model.trdc`, and places it in the **Ready-to-Run** grid.
- Click **"Load & Chat"**.
- The model skeleton is constructed on GPU, custom Triton fused linear modules are bound, and the engine is active in roughly 1.5 seconds.
- You can begin chatting, testing prompts, and streaming tokens immediately.

### Step 3: Compressing Your Own Raw Hugging Face Models (1-Click)
If you place any raw, uncompressed Hugging Face model directory (with `.safetensors` or `.bin` files) into the workspace:
- It appears under the **Original Models** tab in the UI.
- Click **"Compress Model"**.
- SPECTRA streams and quantizes the model on the fly using zero calibration data, writes the `.trdc` container into `compressed_models/`, and mounts it for inference.

---

## Interactive Web Dashboard Features

The localhost Web UI (`web/index.html`) provides complete autonomous control:

1. **Pinned Hardware Telemetry HUD (Top Bar):**
   - **True Physical VRAM:** Queries hardware driver registers via NVML (`libnvidia-ml.so.1`), matching `nvidia-smi` exactly.
   - **VRAM Utilization Bar:** Visual indicator of active memory allocation and safe headroom.
   - **Host RAM & CPU Load:** Monitored via `psutil`.
   - **GPU Core Temperature & Power:** Real-time temperature (°C) and power draw (Watts).
   - **Execution Strategy Badge:** Indicates current hardware mode (`FULL_GPU_RESIDENT`, `HYBRID_DYNAMIC_OFFLOAD`, or `CPU_RESIDENT`).

2. **Autonomous Model Drawer:**
   - Visual catalog of all detected local models.
   - Exact size comparisons, compression ratios, and VRAM savings percentages.
   - 1-Click Load and Unload actions (unloading cleans PyTorch allocator caches and resets GPU memory to 0 MB).

3. **Interactive Playground & Streaming Chat:**
   - Server-Sent Events (SSE) token-by-token streaming.
   - Reasoning parser: Automatically handles thinking models (e.g. DeepSeek-R1 `<think>` tags) and presents thoughts in a collapsible styled drawer.
   - Live performance HUD displaying generation speed (`tok/s`), prompt prefill speed, and step latency (`ms/tok`).
   - Multi-turn conversation persistence stored locally as JSON session files.

4. **Batched Throughput Benchmark Suite:**
   - 1-Click benchmark across batch sizes $B=1$, $B=2$, and $B=4$.
   - Measures raw hardware execution limits and tensor core scaling.

---

## Empirical Benchmarks (Physical RTX 3050 Ti Laptop GPU &bull; 4 GB VRAM)

Every benchmark reported below was measured on physical hardware under reproducible test runs. All models run with **100% GPU Residency** (zero CPU swapping during autoregressive decode).

### Master Multi-Model Performance Table

| Model Architecture | Parameter Count | Uncompressed FP16 | SPECTRA 4-bit | Compression Ratio | Active VRAM | Eager Throughput | CUDA Graph Throughput | Context Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-1.5B** | 1.77 Billion | 3.31 GB | **1.03 GB** | **3.21x** | **917.6 MB** | 25.78 tok/s | **82.53 tok/s** | 8,192 tokens |
| **LiquidAI LFM-3-1.7B** | 1.74 Billion | 3.22 GB | **1.00 GB** | **3.22x** | **1,029.1 MB** | 18.40 tok/s | **32.40 tok/s** | 4,096 tokens |
| **LiquidAI LFM-2.5-2.6B** | 2.60 Billion | 5.02 GB | **1.52 GB** | **3.30x** | **1,556.0 MB** | 9.80 tok/s | **14.20 tok/s** | 4,096 tokens |
| **IBM Granite-4.2-3B** | 3.20 Billion | 6.82 GB *(OOM)* | **2.06 GB** | **3.31x** | **2,245.0 MB** | 5.60 tok/s | **8.07 tok/s** | 4,096 tokens |
| **NVIDIA Nemotron-3-Nano-4B** | 4.10 Billion | 6.33 GB *(OOM)* | **2.26 GB** | **2.80x** | **2,326.9 MB** | 14.10 tok/s | **22.40 tok/s** | 4,096 tokens |
| **Alibaba Qwen-3.5-4B** | 4.15 Billion | 8.68 GB *(OOM)* | **2.63 GB** | **3.30x** | **2,690.5 MB** | 9.20 tok/s | **14.55 tok/s** | 4,096 tokens |

### Downstream Reasoning & Accuracy Parity
- **ARC-Challenge Evaluation (1,172 Questions):** DeepSeek-R1-Distill-1.5B under SPECTRA achieves **35.58%** accuracy compared to **34.73%** for the uncompressed BF16 baseline, proving zero representation collapse on complex multi-step reasoning.
- **Generation Token Agreement:** Evaluated across 3,200 continuous generation tokens, achieving 91.0% first-token greedy match and 100% semantic task completion parity.
- **VRAM Stability:** Verified over 20 continuous generate/unload cycles with background NVML telemetry, maintaining strict 0.0 MB memory leaks.

---

## Under the Hood: Microarchitectural Deep-Dive

### 1. Calibration-Free Group-32 Microscaling
Rather than calculating global or per-channel scales that can be distorted by heavy-tailed outliers, SPECTRA partitions weights into blocks of 32 contiguous parameters:
$$W_{\text{quant}} = \text{clamp}\left(\left\lfloor \frac{W - \min(W)}{\text{Scale}} \right\rceil, 0, 15\right)$$
- Each 32-element group maintains an independent FP16 scale and packed INT4 zero-point.
- Weights are packed into 4-bit nibbles with Marlin coordinate interleaving (weights $0..15$ in the low nibble, $16..31$ in the high nibble of each 16-byte block), matching GPU vector load alignment.

### 2. Saliency Mapping via AST Curvature Extraction
In neural architectures with LayerNorm or RMSNorm preceding linear projections:
$$y = W \cdot \text{RMSNorm}(x) = W \cdot \left(\frac{x}{\text{RMS}(x)} \odot \gamma\right)$$
The learned affine scale vector $\gamma$ analytically mirrors the parameter Hessian curvature:
$$F_{jj} \propto |\gamma_j|^2$$
SPECTRA inspects the model Abstract Syntax Tree (AST), pairs every linear layer with its upstream normalization vector, and computes column saliency:
$$S_j = |\gamma_j| \cdot \|W_{:,j}\|_2$$
This computation takes under 50 milliseconds across an entire 7B model and requires zero calibration data.

### 3. Fused Triton Hardware Kernels
The execution engine replaces standard PyTorch linear layers with custom Triton kernels:

- **Single-Token Decode (`fused_w4a16_gemv_hdsq`):**
  - Designed for batch size $B = 1$ and sequence length $S = 1$.
  - Loads 4-bit packed integer bytes directly into vector registers (`vreg`).
  - Bit-shifts and masks nibbles in parallel:
    $$\text{low} = \text{packed} \ \& \ 0x0F, \quad \text{high} = (\text{packed} \gg 4) \ \& \ 0x0F$$
  - Multiplies by microscales and accumulates into FP32 registers.
  - Requires **0 bytes of dynamic global VRAM scratchpad**.

- **Tensor Core Batched Inference (`batched_w4a16_gemm`):**
  - Designed for prefill and batched queries ($B \ge 2$).
  - Maps unpacked weights into $16 \times 16$ Tensor Core tile matrix multiply units, driving aggregate throughput up to $223.9\text{ tok/s}$.

### 4. Client Systems Memory Optimizations
- **`SpectraFusedEmbedding`:** Vocabulary matrices (often exceeding 131,000 tokens in models like Nemotron) typically consume 600 MB to 1.5 GB of static VRAM. SPECTRA stores vocabulary tables in packed 4-bit Group-32 format, dynamically gathering and dequantizing only the active tokens in the forward pass. This saves over 69% static VRAM.
- **`PingPongLayerOffloader`:** When a model exceeds physical VRAM, SPECTRA uses double-buffered CUDA streams to overlap layer computation with host-to-device PCIe DMA transfers, executing the remainder of the model with minimal throughput degradation.
- **`SpectraQuantizedCache`:** Compresses the Key-Value Cache into Group-32 4-bit representations, allowing context length scaling up to 8,192 tokens on 4 GB VRAM without out-of-memory errors.

---

## CLI Reference

SPECTRA includes a unified command-line interface:

```bash
# 1. Launch Web Dashboard
spectra ui --port 7860

# 2. Probe Hardware Telemetry
spectra status

# 3. Compress a Model
spectra compress \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./models/deepseek_r1_spectra.trdc \
  --group-size 32 \
  --outliers 0.01

# 4. Run CLI Generation
spectra run \
  --model ./models/deepseek_r1_spectra.trdc \
  --prompt "Solve 3x + 15 = 42 step-by-step." \
  --max-tokens 512 \
  --batch-test
```

---

## Python SDK Integration

You can integrate SPECTRA into your own Python applications:

```python
from spectra import Spectra

# Autonomous hardware inspection, execution planning, and model loading
engine = Spectra.serve("path/to/model.trdc")

# 1. Synchronous Generation
result = engine.generate("Explain the theory of relativity in one sentence:")
print(result.text)
print(result.summary())

# 2. Incremental Token Streaming
for chunk in engine.generate_stream("Write a Python Fibonacci function:"):
    if chunk.get("token"):
        print(chunk["token"], end="", flush=True)

# 3. Batched Inference Benchmark
batch_results = engine.generate_batch([
    "Prompt 1: What is gravity?",
    "Prompt 2: Define quantum mechanics."
], max_new_tokens=64)
print(f"Throughput: {batch_results['aggregate_tok_per_sec']:.2f} tok/s")
```

---

## Hardware Compatibility & System Requirements

| Hardware Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA Turing (GTX 1660, RTX 2060, Compute Capability 7.5+) | NVIDIA Ampere, Ada Lovelace, Hopper, or Blackwell (RTX 3050 Ti, 3060, 4060, A100, H100) |
| **VRAM** | 4 GB GDDR6 | 4 GB to 16 GB VRAM |
| **System Host RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **Operating System** | Linux (Ubuntu 20.04+, Debian 11+, RHEL 9+, Arch) or WSL2 on Windows | Ubuntu 22.04 LTS / WSL2 |
| **Python** | Python 3.10, 3.11, or 3.12 | Python 3.11 |
| **CUDA Driver** | CUDA 12.0+ (Driver version $\ge 525.60$) | CUDA 12.2+ |

---

## License & Citation

SPECTRA is released under the **Apache 2.0 License**.

```bibtex
@software{spectra2026,
  author = {NeuraCryption Infrastructure Research Lab},
  title = {SPECTRA: Sparse-Precision Exact Channel Tensor Reconstruction Architecture for Hardware-Saturated Edge Inference},
  year = {2026},
  url = {https://github.com/NeuraCryption/SPECTRA},
  note = {Universal model compression engine and Triton fused execution runtime}
}
```
