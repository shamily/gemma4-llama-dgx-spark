# llama_host

Running and benchmarking LLMs with llama.cpp in Docker on the ASUS Ascent GX10 (NVIDIA DGX Spark).

## Hardware

| | |
|---|---|
| Platform | ASUS Ascent GX10 / NVIDIA DGX Spark |
| SoC | NVIDIA GB10 Grace Blackwell |
| Architecture | ARM64 (aarch64) |
| Memory | 128 GB unified (CPU + GPU share one pool) |
| CUDA | 13.0 (SM_121) |

## Model

| | |
|---|---|
| Model | `tianrui6641/gemma-4-26b-a4b-gguf-mxfp4-moe` |
| Base | `google/gemma-4-26B-A4B` (base + embedded chat template) |
| File | `gemma-4-26B-A4B.MXFP4_MOE.gguf` (14.7 GB) |
| Projector | `mmproj-gemma-4-26B-A4B.f16.gguf` (vision) |
| Architecture | Gemma 4 (MoE + hybrid sliding-window attention) |
| Active params | ~4B per token (26B total) |
| Quantization | MXFP4_MOE — 4-bit MoE-optimized |
| License | Apache 2.0 — public, no HF token needed |
| llama.cpp support | PR [#21309](https://github.com/ggml-org/llama.cpp/pull/21309) — vision + MoE (audio not yet supported) |

## Key Technical Notes

- **Docker base image:** `nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04` (from NGC, not Docker Hub — only NGC has ARM64 CUDA 13 images)
- **Compute capability:** SM_121 (GB10 Grace Blackwell SoC). **Not** SM_100 (discrete Blackwell RTX). If SM_121 build fails, fall back to `CUDA_DOCKER_ARCH=89` (Ada PTX — works via forward compatibility)
- **Unified memory:** no PCIe bottleneck; 128 GB available to llama.cpp with `--n-gpu-layers -1`

## Project Structure

```
.
├── Dockerfile              # llama.cpp CUDA build from source (SM_121, CUDA 13)
├── docker-compose.yml      # server + bench services
├── scripts/
│   ├── download_model.sh   # HuggingFace download → GGUF convert → quantize
│   └── run_bench.sh        # llama-bench + llama-batched-bench (DGX Spark methodology)
├── models/                 # GGUF model files (gitignored)
└── results/                # Benchmark output markdown files
```

## Quickstart

### 1. Build the image

```bash
docker compose build
```

### 2. Download & convert the model

```bash
HF_TOKEN=hf_xxx ./scripts/download_model.sh
```

Gemma 4 is a gated model — accept the license at [huggingface.co/google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) first.

### 3. Start the server

```bash
docker compose up server
```

API available at `http://localhost:8080` (OpenAI-compatible).

### 4. Run benchmarks

```bash
./scripts/run_bench.sh
```

Results saved to `./results/bench_<timestamp>.md`.

## Planned Work

- [x] Dockerfile — llama.cpp CUDA image (SM_121, CUDA 13, ARM64)
- [x] docker-compose.yml — server + bench services
- [x] Model download / conversion script (HuggingFace → GGUF + quantize)
- [x] Benchmark script (llama-bench + llama-batched-bench)
- [ ] Inference smoke test script
- [ ] Results analysis & visualization
- [ ] Article writeup

## Changelog

### 2026-04-05
- Initialized repo
- Confirmed Gemma 4 support in llama.cpp (PR #21309)
- Confirmed hardware: ASUS Ascent GX10 (DGX Spark) — GB10, SM_121, ARM64, CUDA 13, 128GB unified memory
- Created Dockerfile (nvcr.io NGC base, CUDA 13.0.1, SM_121, builds llama.cpp from source)
- Created docker-compose.yml (server + bench profiles)
- Created scripts: download_model.sh, run_bench.sh, test_inference.sh
- Build succeeded: llama.cpp compiled with CUDA SM_121, BLACKWELL_NATIVE_FP4=1
- Server running: gemma-4-26B-A4B-it Q4_K_M (ggml-org), 31/31 layers on GPU, ~69 t/s generation
- Model has built-in chain-of-thought reasoning (reasoning_content in responses)
