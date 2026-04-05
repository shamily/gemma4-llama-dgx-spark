# llama_host

A reproducible Docker setup for running and benchmarking **Google Gemma 4 26B** with llama.cpp on the NVIDIA GB10 (Grace Blackwell) SoC. Includes a working OpenAI-compatible inference server, benchmark scripts, and real results.

> This repo targets the ASUS Ascent GX10 / NVIDIA DGX Spark hardware specifically because of its unusual ARM64 + unified memory architecture. If you have a standard x86 NVIDIA GPU, the Dockerfile will work with minor changes to the base image and `CUDA_DOCKER_ARCH`.

---

## What is this model?

**Gemma 4 26B** is Google's latest open-weights model. It uses a **Mixture of Experts (MoE)** architecture — meaning it has 26 billion total parameters, but only ~4 billion are active on any given token. The inactive experts stay dormant. This makes it faster and cheaper to run than a dense 26B model while retaining comparable quality.

The `-it` suffix means **instruction-tuned**: fine-tuned by Google to follow instructions and hold conversations. The base model (without `-it`) just continues text and is not suitable for chat.

Gemma 4 also has **built-in chain-of-thought reasoning** — before generating a response, the model silently thinks through the problem. The API returns this as a separate `reasoning_content` field alongside the final `content`.

---

## Hardware

| | |
|---|---|
| Platform | ASUS Ascent GX10 (consumer name) = NVIDIA DGX Spark (enterprise name) — same device |
| SoC | NVIDIA GB10 Grace Blackwell |
| Architecture | ARM64 (aarch64) — not x86 |
| Memory | 128 GB unified — CPU and GPU share one physical memory pool, no PCIe transfers |
| CUDA | 13.0, compute capability SM_121 |

**Why unified memory matters:** on a standard GPU, the model weights must be copied over PCIe from system RAM to GPU VRAM. Here there is no copy — the full 128 GB is directly addressable by both CPU and GPU. The 16 GB model loads and runs entirely on-chip.

**SM_121 vs SM_100:** discrete Blackwell RTX cards (e.g. RTX 5090) are SM_100. The GB10 SoC is SM_121 — a different chip. Many build guides for "Blackwell" are wrong for this hardware. This repo targets SM_121 explicitly.

---

## Model

| | |
|---|---|
| Source | `ggml-org/gemma-4-26B-A4B-it-GGUF` on HuggingFace |
| Base weights | `google/gemma-4-26B-A4B-it` — Google's official release |
| File | `gemma-4-26B-A4B-it-Q4_K_M.gguf` (16 GB) |
| Quantization | Q4_K_M — 4-bit mixed precision. Chosen for GB10: fits in ~16 GB, leaves headroom for KV cache, and the hardware's native FP4 support means minimal quality loss vs higher-bit quants |
| License | Apache 2.0 — no HF token needed, no gating |
| llama.cpp support | Added in PR [#21309](https://github.com/ggml-org/llama.cpp/pull/21309) — vision + MoE supported; audio not yet |

**Why ggml-org and not other GGUF sources?** `ggml-org` is the team that builds llama.cpp. They convert directly from Google's official safetensors weights. Google does not publish GGUF files themselves.

---

## Prerequisites

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured
- ~40 GB free disk space (16 GB model + Docker image layers)
- NVIDIA driver 580+ (CUDA 13 support)

Verify your setup:
```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

---

## Quickstart

### 1. Build the image

Clones llama.cpp from source and compiles with CUDA SM_121 support. Takes ~3 minutes on GB10.

```bash
docker compose build
```

### 2. Download the model

No HF token required — the model is public.

```bash
./scripts/download_model.sh
```

### 3. Start the inference server

```bash
docker compose up server
```

OpenAI-compatible API available at `http://localhost:8080`.

### 4. Test it

```bash
./scripts/test_inference.sh
```

Expected output includes a `[thinking]` block (Gemma 4's chain-of-thought) followed by the answer, plus prompt/generation speed.

### 5. Run benchmarks

```bash
./scripts/run_bench.sh
```

Results saved to `./results/bench_<timestamp>.md`.

---

## Using the API

The server is OpenAI-compatible. Use the OpenAI Python SDK pointed at localhost:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="gemma-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
# The capital of France is Paris.
```

**Accessing Gemma 4's reasoning:**

```python
# The model thinks before it answers — access the thought process:
thinking = response.choices[0].message.model_extra.get("reasoning_content")
answer = response.choices[0].message.content
```

If you already use the OpenAI API in your code, change one line:
```python
# Before
client = OpenAI(api_key="sk-...")
# After — no other changes
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
```

---

## Benchmark Results

**Hardware:** NVIDIA GB10, SM_121, 120 GB unified memory, ARM64
**llama.cpp:** build `5d3a4a7`, CUDA 13, flash attention enabled, `BLACKWELL_NATIVE_FP4=1`

`BLACKWELL_NATIVE_FP4=1` means the GB10 executes 4-bit operations in hardware — no software emulation. This is why Q4_K_M performs well on this chip specifically.

### Single-sequence throughput (llama-bench)

Model: `gemma-4-26B-A4B-it Q4_K_M`, 15.63 GiB, 25.23 B params

| test | t/s |
|---|---|
| pp512 — prompt processing (prefill) | **2607 t/s** |
| tg128 — token generation | **70 t/s** |

- **Prompt processing (pp):** how fast the model ingests your input. 2607 t/s means a 2600-token prompt is processed in about 1 second.
- **Token generation (tg):** how fast it produces output. 70 t/s is ~70 words/second — fast enough that streaming feels instant.

For context: a dense 26B model on a high-end x86 workstation GPU typically runs 20–40 t/s generation. The GB10's unified memory and native FP4 push this to 70 t/s.

### Multi-sequence throughput (llama-batched-bench)

How fast the server handles multiple parallel requests. B = number of simultaneous sequences.

| PP | TG | B | N_KV | S_PP t/s | S_TG t/s | S total t/s |
|---|---|---|---|---|---|---|
| 128 | 32 | 1 | 160 | 461 | 58 | 193 |
| 128 | 32 | 4 | 640 | 2727 | 160 | 649 |
| 128 | 32 | 8 | 1280 | 3123 | 228 | 881 |
| 512 | 32 | 1 | 544 | 2740 | 57 | 728 |
| 512 | 32 | 4 | 2176 | 3233 | 147 | 1445 |
| 512 | 32 | 8 | 4352 | 3278 | 201 | 1725 |
| 512 | 128 | 1 | 640 | 2814 | 58 | 269 |
| 512 | 128 | 4 | 2560 | 3257 | 148 | 627 |
| 512 | 128 | 8 | 5120 | 3288 | 204 | 816 |

At B=8 parallel sequences the prefill throughput reaches **3288 t/s** — batching is efficient because unified memory eliminates the bottleneck that limits batched inference on discrete GPUs.

Full benchmark logs in [`results/`](./results/).

---

## Project Structure

```
.
├── Dockerfile              # llama.cpp built from source, CUDA 13, SM_121, ARM64
├── docker-compose.yml      # server service (port 8080) + bench profile
├── scripts/
│   ├── download_model.sh   # downloads Q4_K_M GGUF from ggml-org
│   ├── test_inference.sh   # smoke test: health check + sample prompt + speed
│   └── run_bench.sh        # llama-bench + llama-batched-bench, saves to results/
├── models/                 # GGUF model files — gitignored, not committed
└── results/                # benchmark output markdown files
```

---

## Changelog

### 2026-04-05
- Confirmed Gemma 4 llama.cpp support (PR #21309 merged)
- Dockerfile: NGC base image (`nvcr.io/nvidia/cuda:13.0.1`), SM_121, ARM64 — build succeeds in ~3 min
- Server: 31/31 layers offloaded to GPU, `BLACKWELL_NATIVE_FP4=1` confirmed
- Model: `ggml-org/gemma-4-26B-A4B-it-GGUF` Q4_K_M — 70 t/s generation, chain-of-thought reasoning active
- Benchmarks: pp512 = 2607 t/s, tg128 = 70 t/s; batched prefill peaks at 3288 t/s (B=8)
