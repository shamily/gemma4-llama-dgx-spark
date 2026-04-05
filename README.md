# Setup and Benchmarking of Gemma 4 on NVIDIA DGX Spark / Asus Ascent GX10 with Nvidia GB10 GPU

Benchmarking **Google Gemma 4 26B** with llama.cpp on the NVIDIA GB10 (Grace Blackwell) SoC, served via a Docker container with an OpenAI-compatible API.

> **Hardware-specific:** This repo targets the ASUS Ascent GX10 / NVIDIA DGX Spark (same device, two names). The ARM64 architecture and unified memory make the build non-obvious. If you have a standard x86 NVIDIA GPU: change the base image from `nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04` to the Docker Hub equivalent for your CUDA version, and set `CUDA_DOCKER_ARCH` to your GPU's compute capability (e.g. `89` for RTX 4090, `90` for H100).

---

## What is this model?

**Gemma 4 26B** is a Google open-weights model (released April 2025). It uses a **Mixture of Experts (MoE)** architecture: 25.23 billion total parameters, but only ~3.8 billion are active on any given token. Inactive experts stay idle. This makes it cheaper to run than a dense 26B model while retaining comparable quality.

The **`-it`** suffix means **instruction-tuned**: fine-tuned by Google to follow instructions and hold conversations. The base model (without `-it`) predicts the next token without any instruction-following behaviour and is not suitable for chat.

Gemma 4 also has **built-in chain-of-thought reasoning** — before generating a response, the model silently thinks through the problem. The API returns this thinking as a separate `reasoning_content` field alongside the final `content`.

---

## Hardware

| | |
|---|---|
| Platform | ASUS Ascent GX10 = NVIDIA DGX Spark — same device, consumer vs enterprise branding |
| SoC | NVIDIA GB10 Grace Blackwell |
| Architecture | ARM64 (aarch64) — not x86 |
| Memory | 128 GB unified — CPU and GPU share one physical memory pool, no PCIe bus between them |
| CUDA | 13.0, compute capability SM_121 |

**Why unified memory matters:** on a standard discrete GPU, model weights sit in VRAM (e.g. 24 GB on RTX 4090). If the model is larger than VRAM, excess layers spill over PCIe to system RAM — which is slow. On the GB10, all 128 GB is one pool. The entire 16 GB model and its KV cache live in the same memory the CPU sees, with no transfer overhead.

**SM_121, not SM_100:** discrete Blackwell RTX cards (RTX 5090 etc.) are SM_100. The GB10 SoC is SM_121 — a different chip with different CUDA kernel requirements. Build guides written for "Blackwell" often target SM_100 and will produce incorrect binaries for this hardware. This repo uses `CUDA_DOCKER_ARCH=121` explicitly.

---

## Model

| | |
|---|---|
| Source | `ggml-org/gemma-4-26B-A4B-it-GGUF` on HuggingFace |
| Base weights | `google/gemma-4-26B-A4B-it` — Google's official release |
| File | `gemma-4-26B-A4B-it-Q4_K_M.gguf` (16 GB on disk) |
| Quantization | Q4_K_M — 4-bit mixed precision. Chosen because it fits within 16 GB, leaving ~100 GB of unified memory free for the OS, KV cache, and future batching headroom |
| License | Apache 2.0 — public repo, no HF token required |
| llama.cpp support | Added in PR [#21309](https://github.com/ggml-org/llama.cpp/pull/21309) — vision + MoE supported; audio not yet |
| Context window | 8192 tokens (server default; model supports up to 256K — see docker-compose.yml to raise `LLAMA_ARG_CTX_SIZE`) |

**Why ggml-org?** `ggml-org` is the team that builds llama.cpp. They convert directly from Google's safetensors weights. Google does not publish GGUF files.

---

## Prerequisites

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- NVIDIA driver 580+ (required for CUDA 13)
- ~50 GB free disk space: 16 GB model + ~2 GB CUDA dev image + compiled image layers

Verify GPU access from Docker (uses NGC image, same as the build):

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

---

## Quickstart

### 1. Build the image

Pulls the CUDA 13 dev image (~2 GB, once) then compiles llama.cpp from source. Compile step takes ~3 minutes on GB10; total with image pull is longer on first run.

```bash
docker compose build
```

The image pins llama.cpp at commit `5d3a4a7`. To update to a newer commit, rebuild — the Dockerfile pulls `master` at build time.

### 2. Download the model

```bash
./scripts/download_model.sh
```

No HF token required. Downloads ~16 GB.

### 3. Start the inference server

```bash
docker compose up server
```

OpenAI-compatible API at `http://localhost:8080`. Stop with `docker compose down`.

### 4. Test it

```bash
./scripts/test_inference.sh
```

Prints a `[thinking]` block (the model's chain-of-thought, reformatted by the script) then the final answer, plus measured prompt and generation speed.

### 5. Run benchmarks

```bash
./scripts/run_bench.sh
```

Runs `llama-bench` (single-sequence) and `llama-batched-bench` (multi-sequence). Results saved to `./results/bench_<timestamp>.md`.

---

## Using the API

The server is OpenAI-compatible. Use the OpenAI Python SDK pointed at localhost:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
# api_key is required by the SDK but ignored by llama-server
# The model field is also ignored — whatever is loaded is used
response = client.chat.completions.create(
    model="ignored",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
# The capital of France is Paris.
```

**Accessing Gemma 4's reasoning:**

`reasoning_content` is a llama-server extension not in the OpenAI spec. The standard SDK exposes unknown fields via `model_extra`:

```python
msg = response.choices[0].message
thinking = msg.model_extra.get("reasoning_content")  # may be None if model doesn't reason
answer = msg.content
```

If you already use the OpenAI API, swap one line:

```python
# Before
client = OpenAI(api_key="sk-...")
# After
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
```

---

## Benchmark Results

**Hardware:** NVIDIA GB10, SM_121, 128 GB unified memory (122 GB available to CUDA), ARM64
**Model:** `gemma-4-26B-A4B-it Q4_K_M`, 15.63 GiB, 25.23 B params
**llama.cpp:** commit `5d3a4a7`, CUDA 13, flash attention on, batch size 2048

### Single-sequence throughput (llama-bench)

| test | t/s |
|---|---|
| pp512 — prompt processing | **2607 t/s** |
| tg128 — token generation | **70 t/s** |

**pp (prompt processing / prefill):** how fast the model processes your input before it starts generating. 2607 t/s means a 2600-token prompt is fully ingested in about 1 second.

**tg (token generation):** output speed. 70 tokens/s is roughly 50 words/second, fast enough that streaming feels real-time.

For comparison: a 26B Q4_K_M model at 16 GB fits in a single RTX 4090 (24 GB VRAM), where it typically generates at ~25–30 t/s. Dual A100 80 GB runs ~35–40 t/s. The GB10's unified memory means no layers spill to system RAM, which is why a single-chip consumer device reaches 70 t/s.

### Multi-sequence throughput (llama-batched-bench)

How fast the server handles concurrent requests. **B** = number of simultaneous sequences, **PP** = prompt tokens per sequence, **TG** = generation tokens per sequence, **N_KV** = total KV cache tokens consumed.

| PP | TG | B | N_KV | S_PP t/s | S_TG t/s | S total t/s |
|---|---|---|---|---|---|---|
| 128 | 32 | 1 | 160 | 452 | 66 | 209 |
| 128 | 32 | 4 | 640 | 2611 | 167 | 665 |
| 128 | 32 | 8 | 1280 | 2842 | 229 | 866 |
| 128 | 128 | 1 | 256 | 1295 | 69 | 132 |
| 128 | 128 | 4 | 1024 | 2614 | 171 | 321 |
| 128 | 128 | 8 | 2048 | 2876 | 234 | 434 |
| 512 | 32 | 1 | 544 | 2601 | 66 | 794 |
| 512 | 32 | 4 | 2176 | 2966 | 156 | 1440 |
| 512 | 32 | 8 | 4352 | 2965 | 205 | 1656 |
| 512 | 128 | 1 | 640 | 2683 | 67 | 304 |
| 512 | 128 | 4 | 2560 | 2939 | 156 | 643 |
| 512 | 128 | 8 | 5120 | 2962 | 207 | 809 |

Prefill scales well with batch size: B=1 gives ~2600 t/s, B=8 gives ~2960 t/s. This is because the GB10 has enough memory bandwidth to keep all 8 sequences' KV caches in the same unified pool without any spilling.

Full benchmark logs in [`results/`](./results/).

---

## Dockerfile Details & Explanations

The `Dockerfile` in this repository is explicitly configured for the unique ARM64 + GB10 environment:

- **NGC Base Images:** Uses NVIDIA GPU Cloud (`nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04`) instead of standard Docker Hub images because Docker Hub lacks official ARM64 CUDA 13 images.
- **Compute Capability SM_121:** Automatically sets `CUDA_DOCKER_ARCH=121` (mapped to `CMAKE_CUDA_ARCHITECTURES=121`). The GB10 Grace Blackwell SoC is SM_121, distinct from discrete Blackwell GPUs (SM_100). Default builds targeting generic Blackwell will fail or generate incompatible PTX. 
- **Compiled From Source:** Clones `ggml-org/llama.cpp` (defaulting to the `master` branch) and compiles via GCC-14 and CMake. This is necessary to immediately support cutting-edge architecture patches like PR #21309 for Gemma 4.
- **Minimal Multi-stage Build:** Compiles the software in a comprehensive `-devel` CUDA image, then copies only the necessary compiled binaries, Python scripts, and `.so` shared libraries into a lightweight `-runtime` image.
- **Optimized Defaults:** Pre-configures the `llama-server` entrypoint with vital environment defaults out of the box, including Flash Attention enabled (`LLAMA_ARG_FLASH_ATTN=1`), context size `8192`, and full GPU memory offloading (`LLAMA_ARG_N_GPU_LAYERS=-1`).

---

## Project Structure

```
.
├── Dockerfile              # llama.cpp built from source, CUDA 13, SM_121, ARM64
├── docker-compose.yml      # server service (port 8080) + bench profile
├── scripts/
│   ├── download_model.sh   # downloads Q4_K_M GGUF from ggml-org (~16 GB)
│   ├── test_inference.sh   # health check + sample prompt + speed readout
│   └── run_bench.sh        # llama-bench + llama-batched-bench, saves to results/
├── models/                 # GGUF model files — gitignored, not committed
└── results/                # benchmark output — not gitignored, committed as evidence
```

---

## Changelog

### 2026-04-05
- Confirmed Gemma 4 llama.cpp support (PR #21309 merged)
- Dockerfile: NGC base image (`nvcr.io/nvidia/cuda:13.0.1`), SM_121, ARM64 — builds in ~3 min
- Server: 31/31 layers on GPU, 122 GB VRAM available, chain-of-thought reasoning active
- Model: `ggml-org/gemma-4-26B-A4B-it-GGUF` Q4_K_M — 70 t/s generation
- Benchmarks (llama.cpp `5d3a4a7`): pp512 = 2607 t/s, tg128 = 70 t/s; batched prefill peaks at 2962 t/s (B=8, PP=512)
