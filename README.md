# Inkling-2-Mini / Derisked — Single RTX 3090 (`sm_86`) Kernel & Concurrency Turbo Recipe

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GPU: RTX 3090 24GB](https://img.shields.io/badge/GPU-1%C3%97%20RTX%203090%20(24%20GB)-76B900.svg)](#improvement-metrics)
[![Context: 128K BF16](https://img.shields.io/badge/Context-131%2C072%20(BF16%20KV)-purple.svg)](#improvement-metrics)
[![Engine: vLLM + DSpark](https://img.shields.io/badge/vLLM-DSpark%20Speculative-orange.svg)](#quick-start)

Drop-in Triton/vLLM kernel overlays and concurrency tuning on top of [`lulakien/inkling2-mini-3090-recipe`](https://github.com/lulakien/inkling2-mini-3090-recipe) (`@ 75761b0e`) for serving **`thinkingmachines/Inkling-2-Mini`** and **`Blackfrost-AI/Inkling-2-Mini-Derisked`** at **128K context (`131,072` tokens, BF16 KV cache)** on a **single NVIDIA RTX 3090 (`sm_86`, 24 GB VRAM)**.

```bash
git clone https://github.com/Avyukth/inkling2-mini-3090-turbo && cd inkling2-mini-3090-turbo
MODEL_DIR="$HOME/models/Inkling-2-Mini-Derisked" bash scripts/setup_and_serve.sh
```

---

## What This Optimizes

Even after porting Inkling-2-Mini's Hopper/Blackwell `vllm/vllm-openai:inkling-2-mini` container to Ampere (`sm_86`), three bottlenecks remain on a single 24 GB RTX 3090:

1. **Split-K + `_reduce` launch overhead in `w8_gemm.py`:** Whenever `ceil(N / 64) < 164` (`N < 10,496`), the stock `sm_86` W8 Triton selector forces `SPLIT_K = 2..4`, allocating a temporary FP32 partial tensor and launching a second `_reduce` kernel on **every** `qkvr` projection (`48×/step`) and draft projection (`536 GB/s`, `24.4 µs` on `qkvr`).
2. **Unquantized attention output writer (`wo_ud`):** `inkling2_attention.py` leaves `self.wo_ud` (`[2304, 3072] × 48` logical loop visits = **`680 MB` read per decode step**) in BF16 cuBLAS (`22.9 µs`, `1.10 ms/step`) and holds `170 MiB` of extra static VRAM.
3. **4-sequence concurrency cap (`MAXSEQS=4`):** Stock `1gpu.sh` (`128k`) caps concurrent sequences at `4`, causing `c=8` bursts to queue (`1,232 ms` avg / `1,983 ms` tail TTFT).

### Changes Included
- **Single-pass `sm_86` W8 tiles (`VLLM_W8_FAST=1` in `patches/w8_gemm.py`):** Uses `SPLIT_K=1` tiles (`BN=32, BK=128` for `M≤11`; `BN=128, BK=128, warps=8` for `M≥12`), eliminating **96 `_reduce` kernel launches per decode step**, raising `qkvr` to **`702–719 GB/s`** (`18.7 µs`) and `markov` to **`825 GB/s`** (**98.6%** of measured `836.5 GB/s` GDDR6X bandwidth).
- **W8 `wo_ud` (`W8="attn,wo,head,markov,draft"` in `patches/inkling2_attention.py`):** Cuts `wo_ud` from `22.9 µs` (BF16 cuBLAS) to **`11.1 µs` (`0.53 ms/step` — 2.06× faster)** with in-place output buffer support and frees `170 MiB` VRAM.
- **Doubled concurrent slots (`MAXSEQS=8`) + `FAR_ATTN=1`:** Fits inside the `131,072` BF16 KV budget on 24 GB VRAM, cutting `c=8` TTFT by **59%** and raising `c=8` aggregate throughput by **+50.5%**.

---

## Improvement Metrics

*Hardware: 1× NVIDIA GeForce RTX 3090 (`sm_86`, 82 SMs, 24 GB GDDR6X, stock P2 compute clocks = **836.5 GB/s** measured D2D copy bandwidth, **65.2 TFLOP/s** measured `4096³` BF16 GEMM, 350 W power limit, headless). Model: `Blackfrost-AI/Inkling-2-Mini-Derisked` (`131,072` max context, BF16 KV, DSpark `k=7` adaptive schedule).*

### 1. Single-Stream Decode & Step Time (`spec_bench.py --ignore-eos`, 1,500 tokens)

*Step times are derived from Prometheus `/metrics` (`vllm:spec_decode_num_drafts_total`). CUDA-graphed kernel microbenchmarks predicted **-0.91 ms/step**; end-to-end serving measured **-0.93 ms/step** average across all 6/6 conditions:*

| Workload | Baseline (`1gpu.sh` default) | Turbo Recipe (`W8+=wo, W8_FAST=1`) | Step Time Delta | Decode Speedup |
|---|---|---|---|---|
| **`code` (`temp=0.0`)** | `244.4 tok/s` · `18.43 ms/step` | **`284.5 tok/s`** · **`17.64 ms/step`** | `-0.79 ms/step` (-4.3%) | **+16.4%** |
| **`explain` (`temp=0.0`)** | `164.9 tok/s` · `18.22 ms/step` | **`187.1 tok/s`** · **`17.27 ms/step`** | `-0.95 ms/step` (-5.2%) | **+13.5%** |
| **`story` (`temp=0.0`)** | `100.8 tok/s` · `17.66 ms/step` | **`157.4 tok/s`** · **`16.81 ms/step`** | `-0.85 ms/step` (-4.8%) | **+56.2%** |
| **`code` (`temp=0.6`)** | `219.2 tok/s` · `18.69 ms/step` | **`258.1 tok/s`** · **`17.72 ms/step`** | `-0.97 ms/step` (-5.2%) | **+17.7%** |
| **`explain` (`temp=0.6`)** | `169.2 tok/s` · `18.70 ms/step` | **`201.6 tok/s`** · **`17.68 ms/step`** | `-1.02 ms/step` (-5.5%) | **+19.1%** |
| **`story` (`temp=0.6`)** | `117.2 tok/s` · `17.95 ms/step` | **`124.7 tok/s`** · **`16.98 ms/step`** | `-0.97 ms/step` (-5.4%) | **+6.4%** |

### 2. Multi-Stream, TTFT & Long-Context Summary (`131,072` BF16 Context)

| Metric | Stock `1gpu.sh` (`128k`) | Config-Only (`MAXSEQS=8, FAR_ATTN=1`) | Turbo Recipe (`+ W8+=wo, W8_FAST=1`) | Delta vs. Stock |
|---|---:|---:|---:|---|
| **`c=1` Decode (median of 6 prompts, `temp=0.6`)** | `132.0 tok/s` | `132.0 tok/s` | **`148.4 / 148.6 tok/s`** *(2 boots)* | **+12.5%** *(won 6/6 prompts)* |
| **`c=4` Aggregate Decode (`temp=0.6`)** | `292.0 tok/s` | `292.3 tok/s` | **`301.3–313.9 tok/s`** | **+3.2% to +7.5%** |
| **`c=8` Aggregate (`ignore_eos=True`, 256t×8)** | `354.3 tok/s` | `464.9 tok/s` | **`533.1 tok/s`** | **+50.5%** |
| **`c=8` Avg TTFT (8 concurrent cold ~210t)** | `1,232 ms` | `914.8 ms` | **`504.2 ms`** | **-59.1%** |
| **32-Prompt Paired Step Time (`temp=0.0`, median)** | `18.00 ms/step` | `18.00 ms/step` | **`17.12 ms/step`** | **-0.88 ms/step (-4.9%, won 32/32)** |
| **32-Prompt Paired Mean Decode (`temp=0.0`, 384t)** | `169.4 tok/s` | `169.4 tok/s` | **`180.8 tok/s`** | **+6.7%** |
| **Cold Prefill ~4K / ~16K (BF16)** | `4,364 / 4,402 tok/s` | `4,362 / 4,389 tok/s` | **`4,351 / 4,396 tok/s`** | ~73% of BF16 GEMM peak |
| **Long-Context Retrieval (`51,233` tokens)** | — | `3/3` (`3,847 tok/s`) | **`3/3` exact (`3,851 tok/s` prefill, `234.2 tok/s` decode)** | **Exact retrieval** |
| **20-Task Accuracy Check (math/code/logic)** | `20/20 (100%)` | `20/20 (100%)` | **`20/20 (100%)`** | **Zero regression** |

### 3. CUDA-Graphed Kernel Timings (`torch.cuda.CUDAGraph` + CUDA Events)

| Projection (Calls / Step) | Shape `(M, N, K)` | Stock Kernel | Turbo Kernel (`VLLM_W8_FAST=1` + `W8+=wo`) | Saved / Step |
|---|---|---:|---:|---:|
| **`qkvr` (`×48`)** (`c=1`) | `(8, 5632, 2304)` | `24.4 µs` (`536 GB/s`, `1.17 ms`) | **`18.7 µs`** (`702 GB/s`, `0.90 ms`) `[BN=32,BK=128,SP=1]` | **`-0.27 ms`** |
| **`qkvr` (`×48`)** (`c=4`) | `(12, 5632, 2304)` | `25.5 µs` (`517 GB/s`, `1.22 ms`) | **`18.3 µs`** (`719 GB/s`, `0.88 ms`) `[BN=128,BK=128,SP=1,W=8]` | **`-0.34 ms`** |
| **`wo_ud` (`×48`)** (`c=1..8`) | `(8, 2304, 3072)` | `22.9 µs` *(BF16 cuBLAS, `1.10 ms`)* | **`11.1 µs`** (`644 GB/s`, `0.53 ms`) `[BN=32,BK=128,SP=1,ST=5]` | **`-0.57 ms`** *(+170 MiB VRAM)* |
| **`markov` (`×7`)** | `(1, 201088, 256)` | `68.5 µs` (`758 GB/s`, `0.48 ms`) | **`62.9 µs`** (`825 GB/s`, `0.44 ms`) `[BN=32,BK=128,SP=1,ST=2]` | **`-0.04 ms`** |
| **`draft_qkv` + `draft_o` (`×3`)** | `(8, 5120/2304, ...)` | `22.2 + 14.5 µs` (`0.11 ms`) | **`16.4 + 10.8 µs`** (`726 / 662 GB/s`, `0.08 ms`) | **`-0.03 ms`** |
| **Total Predicted Savings** | | | | **`-0.91 ms / step`** |

### 4. Real-World Agentic Practicality & Cross-Engine Comparison (`1× RTX 3090`)

Measured with [`scripts/practical_bench.py`](scripts/practical_bench.py) ([`results/practical_bench.json`](results/practical_bench.json)) and compared against published single-RTX 3090 figures for `llama.cpp` and `maya-ik2` (`UD-Q4_K_XL`, FP8 KV, single-stream):

| Practical Dimension | `llama.cpp` (GGUF) | `maya-ik2` (`UD-Q4_K_XL`, FP8 KV, no MTP) | **This Recipe (`vLLM` + Turbo Patches, Native NVFP4, BF16 KV + DSpark)** |
|---|---|---|---|
| **Weight & KV Fidelity** | Re-quantized GGUF | Re-quantized `UD-Q4_K_XL` + FP8 KV | **Original native NVFP4 bytes + lossless BF16 KV** *(zero double-quant)* |
| **Decode: Short / 4K / 64K / 124K** | `57.0` / — / `42.7` / — `tok/s` | `129.5` / `124.2` / `105.6` / `90.6 tok/s` | **Prose:** `141.9–187.1` / `132.5` / `115.7` / `95.8 tok/s`<br>**Code:** **`233.4–284.5` / `180.0` / `154.9` / `132.4 tok/s`** |
| **Cold Prefill: 4K / 64K / 124K** | `1,052` / `630` / — `tok/s` | `2,742` / `2,361` / `2,079 tok/s` | **`4,319–4,396` / `3,632` / `3,036 tok/s`** *(+46% to +58% faster)* |
| **Max Single-Card Context** | OOM `< 128K` | **`344K`** *(FP8 KV, CPU vision, no MTP)* | `131,072` *(BF16 KV)* / `262,144` *(via `PROFILE=262k` FP8 KV)* |
| **Multi-Turn `64K` Agent Loop** | ~`102 s` cold; single-slot prompt cache | ~`27.2 s` cold (`2,361 tok/s`) | `17.73 s` cold → **`609 ms` Turns 2–4 (`29.1×` speedup)**; **`750 ms` at `100K` (`40.8×`)** |
| **2 Branched Subagents (`64K` / `100K` shared repo)** | Serialized (re-evaluates / queues) | Serialized (`batch=1` static graph) | **`652 ms` @ `64K` / `807 ms` @ `100K`** *(concurrent paged prefix-cache hit)* |
| **8 Concurrent Streams (`c=8`)** | Queues / severe slowdown | Serialized (`batch=1`) | **`533.1 tok/s` aggregate (`504 ms` avg TTFT)** |
| **Prefill–Decode Interference** | Blocks active stream | Blocks until Request 1 finishes | Stream A keeps **`75.9 tok/s`** while Stream B cold-prefills `32K` in `8.44 s` *(tunable via `LPT=512`)* |
| **`100K` 3-Hop Codebase Synthesis** | >`160 s` wall time | Untested (passkey / GSM8K reported) | **Exact (`270 == 270`) across `10% / 50% / 90%` modules in `33.57 s` total** |
| **Tool & Schema Fidelity (`4` checks)** | Template-dependent | Custom engine | **`4/4` PASS** *(escaped AST `edit_file`, 3-way parallel tools, bug-fix round-trip, strict JSON schema)* |
| **Time-to-Verified-Code (`3` algos, `exec()` tested)** | ~`6–8 s` / task | **Working thinking-off toggle** | **`3/3` PASS in `1.70 s` avg (`effort=low`)** *(note: upstream template treats `effort=none` as default thinking)* |

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Inkling-2-Mini / Derisked (24 physical layers, layers 6..17 looped 3x)     │
│  48 logical layer-visits  |  8 global KV layers + 40 SWA-1024 local layers  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────────┐     ┌─────────────────────┐
│ Attention qkvr    │     │ Attention wo_ud (NEW) │     │ Routed MoE Experts  │
│ W8 single-pass    │     │ BF16 -> W8 single-pass│     │ 128 experts, top-8  │
│ BN=32/128, BK=128 │     │ BN=32, BK=128, ST=5   │     │ Native NVFP4 bytes  │
│ SPLIT_K=1 (no red)│     │ 22.9 us -> 11.1 us    │     │ Humming W4A16 sm_86 │
│ 536 -> 719 GB/s   │     │ Frees 170 MiB VRAM    │     │ ~8.5 ms / step      │
└───────────────────┘     └───────────────────────┘     └─────────────────────┘
          └───────────────────────────┼───────────────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │ DSpark Speculative Drafter (k=7)  │
                    │ Adaptive batch schedule: 7/4/2/1  │
                    │ Markov W8 fold @ 825 GB/s (98.6%) │
                    │ MAXSEQS=8 concurrent sequences    │
                    └───────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Linux x86_64 with an **NVIDIA RTX 3090 (24 GB)**, **no display attached** (the `128k` profile uses ~23.4 GiB of the 24 GiB card)
- NVIDIA driver supporting **CUDA 13+** (tested on `610.x+` / `620.x+`)
- Docker + NVIDIA Container Toolkit (`--gpus all`)
- ~20 GB disk for model weights + ~36 GB disk in Docker storage for the pinned vLLM image

### 1. Download Weights & Pull Pinned Image

```bash
# Either Blackfrost-AI/Inkling-2-Mini-Derisked or thinkingmachines/Inkling-2-Mini
hf download Blackfrost-AI/Inkling-2-Mini-Derisked \
  --local-dir "$HOME/models/Inkling-2-Mini-Derisked" \
  --exclude "DWM_RECIPE/*" --exclude "ASSETS/*"

docker pull vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
```

### 2. Launch the Tuned 128K Server

```bash
git clone https://github.com/Avyukth/inkling2-mini-3090-turbo && cd inkling2-mini-3090-turbo

MODEL_DIR="$HOME/models/Inkling-2-Mini-Derisked" \
SERVED_NAME="Inkling-2-Mini-Derisked" \
HOST="127.0.0.1" \
PORT="8000" \
bash scripts/setup_and_serve.sh
```

`setup_and_serve.sh` clones [`lulakien/inkling2-mini-3090-recipe`](https://github.com/lulakien/inkling2-mini-3090-recipe) at pinned commit `75761b0eaaf2a80aeb76ffa9dff0768300236e2a`, overlays `patches/w8_gemm.py` and `patches/inkling2_attention.py`, launches the container with `PROFILE=128k FAR_ATTN=1 MAXSEQS=8 W8="attn,wo,head,markov,draft" VLLM_W8_FAST=1`, waits for `/health`, and runs batched kernel warm-up.

### 3. Smoke Test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Inkling-2-Mini-Derisked",
    "messages": [{"role": "user", "content": "Write a 1-line Python prime check."}],
    "reasoning_effort": "low",
    "max_tokens": 512
  }'
```

*(Reasoning tokens are returned in `choices[0].message.reasoning` and the final answer in `choices[0].message.content`.)*

### 4. Manual Patch Option (Existing Checkout)

If you already have `lulakien/inkling2-mini-3090-recipe` checked out at `75761b0e`:

```bash
git -C /path/to/inkling2-mini-3090-recipe apply /path/to/inkling2-mini-3090-turbo/patches/sm86_w8_wo_turbo.patch

cd /path/to/inkling2-mini-3090-recipe
MODEL_DIR="$HOME/models/Inkling-2-Mini-Derisked" \
SERVED_NAME="Inkling-2-Mini-Derisked" \
PROFILE="128k" FAR_ATTN=1 MAXSEQS=8 \
W8="attn,wo,head,markov,draft" \
DOCKER_EXTRA="-e VLLM_W8_FAST=1" \
bash launch/1gpu.sh
```

---

## Verification & Benchmarking

```bash
# 1. Standalone numerical equivalence check (stop server first so ~1 GiB VRAM is free)
bash scripts/run_isomorphism_check.sh

# 2. Single-stream step-time + multi-request TTFT & throughput benchmark
python3 scripts/benchmark.py

# 3. 6-part real-world agentic practicality suite (prefix cache, tools, interference, 100K 3-hop, solve time)
python3 scripts/practical_bench.py
```

---

## Tested & Rejected Configurations

| Change Tested | Outcome & Reason Rejected |
|---|---|
| `MNBT=2048` (`--max-num-batched-tokens 2048`) | Prefill warm-up reservation left `5.2 GiB` KV cache vs `5.3 GiB` required for `131,072` BF16 context (max context dropped to `128,128`). |
| `SPEC_SCHED` `c=3..4` draft length `k=2 → k=3` | `c=4` aggregate throughput regressed `292.4 → 288.4 tok/s` (`-1.3%`); extra verify rows outweighed incremental accepted tokens. |
| Humming MoE `use_stream_k=True` | Regressed `c=1` (`130.3 tok/s`) and `c=4` (`289.4 tok/s`), and caused a `4.35 s` JIT compile stall on `c=8` bursts. |
| `VLLM_W8_FAST=1` without `W8+=wo` | Improved `c=8` (`506.8 tok/s`), but required pairing with `W8+=wo` to unlock the full `-0.93 ms/step` (`+12.5%` to `+19.1%`) single-stream gain. |

---

## Repository Layout

```text
patches/
  w8_gemm.py                 # Single-pass sm_86 W8 tile selector (VLLM_W8_FAST=1) + in-place w8_linear(..., out=out)
  inkling2_attention.py      # Optional W8 quantization of self.wo_ud ("wo" in VLLM_INKLING_W8)
  sm86_w8_wo_turbo.patch     # Unified diff against lulakien/inkling2-mini-3090-recipe @ 75761b0e
scripts/
  setup_and_serve.sh         # Clones pinned upstream recipe, installs overlays, launches tuned 128k server
  run_isomorphism_check.sh   # Runs numerical equivalence test inside the pinned vLLM container
  verify_isomorphism.py      # Verifies max_abs_diff <= 1 BF16 ULP across all 6 target GEMM shapes
  benchmark.py               # Measures single-stream ms/step, decode tok/s, and c=1/2/4/8 TTFT & throughput
  practical_bench.py         # 6-part real-world agentic benchmark (multi-turn cache, tools, interference, 100K 3-hop)
results/
  spec_bench_comparison.json # Before vs. After spec_bench.py (1,500 tok, --ignore-eos)
  metrics_summary.json       # Structured summary of single-stream, concurrency, and kernel microbenchmark metrics
  practical_bench.json       # Measured outputs of the 6-part real-world practicality suite
```

---

## Troubleshooting & Notes

- **`ValueError: To serve at least one request with the model's max seq len (131072)...` on boot:** Ensure no desktop session or other CUDA process is occupying VRAM on the RTX 3090 (`nvidia-smi` should report `< 50 MiB` used prior to launch).
- **1M context window:** A single 24 GB RTX 3090 cannot hold 1M context in VRAM (`1,048,576` tokens across the 8 global attention layers requires `32 GiB` in BF16 or `16 GiB` in FP8 on top of `~18.2 GiB` NVFP4 weights). Use `PROFILE=262k` (FP8 KV) for up to `262,144` tokens on 1× 3090, or `2× RTX 3090` (`TP=2`, `PROFILE=1m`) for `1,048,576`.
- **Idle wake-up (`P8 → P2`):** When idle, an RTX 3090 drops to `P8` (`210 MHz` SM / `405 MHz` GDDR6X), adding ~250–300 ms to the first request after idle unless `nvidia-smi -pm 1` is active.

---

## Credits & License

- Built on [`lulakien/inkling2-mini-3090-recipe`](https://github.com/lulakien/inkling2-mini-3090-recipe) (`Apache-2.0`).
- Models: [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and [`Blackfrost-AI/Inkling-2-Mini-Derisked`](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked) (no model weights are distributed in this repository).
- Licensed under [Apache-2.0](LICENSE). Issues and pull requests are welcome.
