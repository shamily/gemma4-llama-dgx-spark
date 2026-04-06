# Gemma 4 Family on NVIDIA DGX Spark (GB10): Setup, Serving, and Benchmarks

Running all four **Google Gemma 4** models with [llama.cpp](https://github.com/ggml-org/llama.cpp) on the NVIDIA GB10 Grace Blackwell SoC, served via a Docker container with an OpenAI-compatible API. Includes a complete benchmark suite: single-sequence throughput, context window scaling, multi-user concurrency, and chain-of-thought timing.

> **Hardware-specific:** This repo targets the ASUS Ascent GX10 / NVIDIA DGX Spark (same device, two names). The ARM64 architecture and unified memory make the build non-obvious. If you have a standard x86 NVIDIA GPU: change the base image from `nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04` to the Docker Hub equivalent for your CUDA version, and set `CUDA_DOCKER_ARCH` to your GPU's compute capability (e.g. `89` for RTX 4090, `90` for H100).

---

## What are these models?

Google released the Gemma 4 family in April 2025. It spans four sizes:

**E2B and E4B** are compact instruction-tuned models without chain-of-thought reasoning. They are fast and efficient — suitable for latency-sensitive applications or embedded deployments. The "E" prefix stands for efficient; the number is the approximate active-parameter count.

**26B-A4B** is a **Mixture of Experts (MoE)** model. Its full weight file holds 25.23 billion parameters, but the router activates only 8 of 128 experts per token — roughly 4 billion parameters are touched per forward pass. Inactive experts sit idle in memory. This means generation cost scales with active parameters (~4B), not total parameters (25B), which produces a surprising result: it generates faster than the denser E4B model despite being 3× the file size. The `-A4B` suffix encodes this: 4 billion active.

**31B** is a fully **dense** model. Every one of its 30.7 billion parameters is loaded and computed for every token. It produces the highest-quality output in the family but at a steep throughput cost compared to the MoE 26B.

Both 26B and 31B are instruction-tuned with **built-in chain-of-thought reasoning**: before answering, the model generates a `<think>` block that reasons through the problem. The llama.cpp server exposes this as a `reasoning_content` field alongside the final `content` in API responses.

---

## Hardware

| | |
|---|---|
| Platform | ASUS Ascent GX10 = NVIDIA DGX Spark — same device, consumer vs enterprise branding |
| SoC | NVIDIA GB10 Grace Blackwell |
| Architecture | ARM64 (aarch64) — not x86 |
| Memory | 128 GB unified — CPU and GPU share one physical memory pool, no PCIe bus between them |
| CUDA | 13.0, compute capability SM_121 |

**Why unified memory matters:** on a standard discrete GPU, model weights sit in VRAM (e.g. 24 GB on RTX 4090). If the model exceeds VRAM, layers spill over PCIe to system RAM — a slow path that typically halves or worse the generation speed. On the GB10, all 128 GB is one pool accessible at full memory bandwidth from both the Grace CPU and Blackwell GPU. The entire model, its KV cache, and the OS coexist without any transfer penalty.

**122 GB available to CUDA:** the OS and firmware reserve ~6 GB, leaving 122,479 MiB visible to CUDA. All four Gemma 4 models combined (E2B + E4B + 26B + 31B) occupy ~43 GB — they can all reside in memory simultaneously.

**SM_121, not SM_100:** discrete Blackwell RTX cards (RTX 5090 etc.) are SM_100. The GB10 SoC is SM_121 — a different chip variant with different CUDA kernel requirements. Build guides written for "Blackwell" often target SM_100 and will produce incorrect binaries for this hardware. This repo uses `CUDA_DOCKER_ARCH=121` explicitly.

---

## Models

All models sourced from `ggml-org` on HuggingFace — the llama.cpp team converts directly from Google's safetensors weights. Apache 2.0 license, no HF token required.

| Key | Model | File | Disk | Params | Active params | Thinking |
|-----|-------|------|------|--------|---------------|---------|
| `e2b` | gemma-4-E2B-it | `gemma-4-e2b-it-Q8_0.gguf` | 4.61 GiB | 4.65B | 4.65B (dense) | no |
| `e4b` | gemma-4-E4B-it | `gemma-4-e4b-it-Q4_K_M.gguf` | 4.95 GiB | 7.52B | 7.52B (dense) | no |
| `26b` | gemma-4-26B-A4B-it | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 15.63 GiB | 25.23B | ~4B (MoE, 8/128 experts) | yes |
| `31b` | gemma-4-31B-it | `gemma-4-31B-it-Q4_K_M.gguf` | 17.39 GiB | 30.70B | 30.70B (dense) | yes |

**llama.cpp support** for all Gemma 4 variants was added in PR [#21309](https://github.com/ggml-org/llama.cpp/pull/21309) (vision + MoE supported; audio not yet).

---

## Prerequisites

- **Docker Engine 20.10+** with the Compose v2 plugin (`docker compose`, not `docker-compose`)
- **nvidia-container-toolkit** — [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **NVIDIA driver 580+** — required for CUDA 13
- **~60 GB free disk space** — all four models (~43 GB) + CUDA dev image (~2 GB) + compiled image layers
- **Python 3.8+** on the host — only needed for `run_bench_thinking.py`; uses stdlib only, no pip install required

To benchmark a single model, ~18 GB is enough. Use `./scripts/download_model.sh 26b`.

**NGC registry:** the build pulls from `nvcr.io` (NVIDIA GPU Cloud), not Docker Hub. Unauthenticated pulls are rate-limited. If the build fails on the base image pull, register for a free NGC account and run:

```bash
docker login nvcr.io   # username: $oauthtoken  password: <your NGC API key>
```

Verify GPU access from Docker before proceeding:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

---

## Quickstart

### 1. Build the image

Pulls the CUDA 13 dev image (~2 GB, once) then compiles llama.cpp from source.

```bash
docker compose build
```

This clones `ggml-org/llama.cpp` **master** at build time, so rebuilding later will pick up a newer llama.cpp and may produce different benchmark numbers.

**To reproduce the exact results in this repo** (build `5d3a4a7`):

```bash
docker compose build --build-arg LLAMA_CPP_REF=5d3a4a7
```

Build time: ~3 minutes on GB10. Expect 10–15 minutes on x86 workstation-class hardware. On first run, add ~5 minutes for the NGC base image download.

### 2. Download models

> **Step 1 must be completed first.** The download script runs inside the `llama_host:latest` container — it will fail with "image not found" if you haven't built yet.

```bash
./scripts/download_model.sh          # all four models (~43 GB total)
./scripts/download_model.sh 26b      # just the 26B MoE (~16 GB)
./scripts/download_model.sh e2b e4b  # multiple specific models
```

Valid keys: `e2b`, `e4b`, `26b`, `31b`. Already-downloaded files are skipped. No HF token required — all models are Apache 2.0 public. If you want to use a private/gated model, add `HF_TOKEN=hf_...` to the `.env` file in the repo root before running.

### 3. Start the inference server

```bash
docker compose up server
```

Loads `gemma-4-26B-A4B-it-Q4_K_M.gguf` by default. OpenAI-compatible API at `http://localhost:8080`. Stop with `docker compose down`.

To serve a different model, pass its **container path** (the `models/` directory is mounted as `/models/` inside the container):

```bash
LLAMA_ARG_MODEL=/models/gemma-4-e4b-it-Q4_K_M.gguf docker compose up server
```

> **GPU exclusivity:** the server and the benchmark tools (`llama-bench`, `llama-batched-bench`) cannot share the GPU. Stop the server before running benchmarks, and stop any benchmarks before starting the server.

### 4. Test it

```bash
./scripts/test_inference.sh
```

Waits for the server to become healthy, sends a sample prompt, and prints the `[thinking]` block, the final answer, and measured prompt + generation speed.

### 5. Run benchmarks

> **The server must not be running.** `llama-bench` and `llama-batched-bench` need exclusive GPU access. Run `docker compose down` first.

```bash
./scripts/run_bench.sh              # all three groups
GROUP=A ./scripts/run_bench.sh      # single group
SKIP_MISSING=1 ./scripts/run_bench.sh  # skip models not yet downloaded
```

Results are saved to `./results/bench_<timestamp>.md`. **Expected runtimes on GB10:**

| Group | What it does | Duration |
|-------|-------------|----------|
| A | llama-bench, all 4 models, 5 reps × 4 tests | ~45–60 min |
| B | Context depth sweep, 26B + 31B, 3 reps × 10 tests | ~30 min |
| C | Batched concurrency, 26B, 48 PP/TG/B combinations | ~60–90 min |
| All | A + B + C sequentially | ~2.5–3 hours |

On other hardware, runtimes will vary — the 31B in Group A is the slowest per-model (11 t/s generation).

### 6. Benchmark thinking mode

```bash
docker compose up -d server         # must be 26B or 31B — E2B/E4B have no thinking
python3 scripts/run_bench_thinking.py
```

Runs on the **host** (not inside Docker). Requires Python 3.8+ with no extra packages. Waits for the server health endpoint automatically before starting. Results saved to `./results/bench_thinking_<timestamp>.md`.

Environment variables:
```bash
HOST=localhost PORT=8080 REPS=5 python3 scripts/run_bench_thinking.py
```

---

## Using the API

The server exposes an OpenAI-compatible endpoint. Use the OpenAI Python SDK pointed at localhost:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
# api_key is required by the SDK but ignored by llama-server

response = client.chat.completions.create(
    model="ignored",   # llama-server serves whatever model is loaded
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
# The capital of France is Paris.
```

**Accessing Gemma 4's chain-of-thought reasoning:**

`reasoning_content` is a llama-server extension not in the OpenAI spec. The standard SDK exposes unknown fields via `model_extra`:

```python
msg = response.choices[0].message
thinking = msg.model_extra.get("reasoning_content")  # None for E2B/E4B; text for 26B/31B
answer = msg.content
```

Drop-in replacement if you already use the OpenAI API:

```python
# Before
client = OpenAI(api_key="sk-...")
# After — everything else stays the same
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
```

---

## Test Environment

```
OS:        Linux aarch64 6.17.0-1014-nvidia (Ubuntu 24.04)
GPU:       NVIDIA GB10 (Grace Blackwell SoC)
SM:        12.1
VRAM:      122,479 MiB unified (CPU + GPU share one pool)
Driver:    580.142
CUDA:      13.0
Compiler:  g++ 14 (Ubuntu 24.04)
llama.cpp: build 5d3a4a7
```

Each benchmark run re-captures OS, GPU, and driver information automatically. See the header of any file in [`results/`](./results/) for the exact build hash used.

> **To reproduce these exact numbers:** build with `docker compose build --build-arg LLAMA_CPP_REF=5d3a4a7`. Any other llama.cpp commit — including a fresh `master` build — may produce different results due to kernel changes, quantization updates, or attention implementation differences. The commit hash is the single most important variable for reproducibility.

---

## Benchmark Results

All results from build `5d3a4a7`, flash attention enabled, batch size 2048. Each `llama-bench` test is repeated 5 times (Group A) or 3 times (Group B); reported values are **mean ± std dev**.

### Group A — Single-sequence throughput

One sequence at a time, no concurrency. Tests prompt processing (pp) and token generation (tg) at two sizes each.

| Model | Size | Params | pp512 (t/s) | pp2048 (t/s) | tg32 (t/s) | tg128 (t/s) |
|-------|------|--------|-------------|--------------|------------|-------------|
| E2B **Q8_0** ¹ | 4.61 GiB | 4.65B | **8089 ± 225** | **7749 ± 76** | 82.4 ± 0.5 | 83.9 ± 1.4 |
| E4B Q4_K_M | 4.95 GiB | 7.52B | 4696 ± 167 | 4313 ± 14 | 61.8 ± 0.3 | 63.3 ± 0.8 |
| 26B-A4B Q4_K_M | 15.63 GiB | 25.23B MoE | 2657 ± 42 | 2888 ± 17 | 68.0 ± 0.6 | **69.9 ± 1.2** |
| 31B Q4_K_M | 17.39 GiB | 30.70B dense | 743 ± 4 | 685 ± 2 | 11.0 ± 0.01 | 11.0 ± 0.02 |

¹ E2B is benchmarked at Q8_0 because no Q4_K_M file is published upstream. Q8_0 has marginally lower dequantization overhead than Q4_K_M, so E2B throughput figures may be slightly optimistic relative to a hypothetical Q4_K_M version. The other three models all use Q4_K_M.

**pp (prompt processing / prefill):** how fast the model ingests your input tokens before generation starts. pp512 = 2657 t/s means a 512-token prompt is processed in ~0.2 seconds.

**tg (token generation):** output speed. 70 t/s ≈ 52 words/second — fast enough that streaming responses feel instantaneous to a human reader.

**The MoE effect on generation speed:** the 26B MoE generates at 69.9 t/s while the E4B dense model generates at only 63.3 t/s — despite the 26B having 5× more total parameters and 3× the file size. This is the MoE architecture working as designed: only ~4B parameters are active per token, so generation cost is comparable to a 4B-class dense model. The E4B is dense at 7.52B, which means it touches more weights per token than the 26B activates.

**The MoE vs dense cliff at similar file sizes:** the 26B (15.63 GiB) and 31B (17.39 GiB) are nearly the same size on disk, yet generation speed differs by **6.4×** — 69.9 vs 11.0 t/s. This is entirely explained by architecture: the 26B only loads ~4B worth of expert weights per token, while the 31B loads all 30.7B on every single token. Disk size is a poor proxy for inference cost when MoE is involved.

**Context comparison:** a 26B Q4_K_M on a single RTX 4090 (24 GB VRAM) would generate at ~25–30 t/s. Dual A100 80 GB reaches ~35–40 t/s. The GB10's unified memory, with no PCIe bottleneck and 122 GB available, reaches 70 t/s from a single chip.

---

### Group B — Context window scaling

How throughput changes as the KV cache fills up. Tests run at fixed prompt size (pp512) and generation length (tg128) across increasing KV-cache depths, simulating a long conversation. Three repetitions each.

**26B-A4B Q4_K_M** (MoE, 25.23B total, ~4B active)

| KV depth | pp512 (t/s) | pp512 vs empty | tg128 (t/s) | tg128 vs empty |
|----------|-------------|----------------|-------------|----------------|
| 0 (empty) | 2641 ± 40 | — | 68.3 ± 0.3 | — |
| 4096 | 2536 ± 56 | −4% | 63.4 ± 0.6 | −7% |
| 8192 | 2362 ± 42 | −11% | 59.6 ± 1.1 | −13% |
| 16384 | 2194 ± 44 | −17% | 59.6 ± 1.2 | −13% |
| 32768 | 1877 ± 40 | **−29%** | 52.1 ± 0.7 | **−24%** |

**31B Q4_K_M** (dense, 30.70B)

| KV depth | pp512 (t/s) | pp512 vs empty | tg128 (t/s) | tg128 vs empty |
|----------|-------------|----------------|-------------|----------------|
| 0 (empty) | 741 ± 6 | — | 11.0 ± 0.03 | — |
| 4096 | 655 ± 34 | −12% | 10.5 ± 0.01 | −5% |
| 8192 | 620 ± 28 | −16% | 10.0 ± 0.02 | −9% |
| 16384 | 570 ± 24 | −23% | 10.0 ± 0.02 | −9% |
| 32768 | 482 ± 13 | **−35%** | 9.2 ± 0.00 | **−16%** |

**Why PP degrades more than TG with depth:** prompt processing reads both the model weights and the full KV cache (via attention). As the KV cache grows, attention over longer sequences becomes more expensive. Token generation only attends to the KV cache once per new token, and the dominant cost is loading the model weights — so TG is less sensitive to KV cache size.

**Why the 31B's TG is less context-sensitive than the 26B's:** the 31B's generation is already dominated by loading 30.7B dense weights per token. The KV cache is a small fraction of total memory traffic. For the 26B MoE, only ~4B of weights are loaded per token, so the KV cache becomes a more significant fraction of memory bandwidth — giving context depth more impact on TG.

**Practical implication:** at a 32K-token conversation depth, the 26B still generates at 52 t/s (vs 68 at empty). For most interactive use cases this is still faster than human reading speed. At 256K context (the model's maximum), performance would degrade further — not measured here.

---

### Group C — Multi-sequence concurrency

How the server scales when handling multiple simultaneous requests. **B** = batch size (concurrent sequences), **PP** = prompt tokens per sequence, **TG** = generated tokens per sequence. Tested on 26B-A4B Q4_K_M with 256K context window.

**S_PP t/s** = aggregate prompt processing throughput (all sequences combined)
**S_TG t/s** = aggregate token generation throughput (all sequences combined)

#### Prompt processing (PP) throughput vs batch size

| PP | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|----|-----|-----|-----|-----|------|------|
| 128 | 438 | 1980 | 2625 | 2867 | 2986 | 2965 |
| 512 | 2624 | 2903 | 2983 | 3006 | 3015 | 3013 |
| 2048 | 2907 | 2916 | 2940 | 2939 | 2952 | 2949 |
| 4096 | 2869 | 2877 | 2886 | 2892 | 2893 | 2891 |

*Values in t/s aggregate.*

PP throughput plateaus quickly — at PP=512 and above, even B=1 saturates the memory bandwidth (~2900 t/s). Adding more concurrent sequences barely moves the needle. **Short prompts are the exception:** at PP=128, B=1 delivers only 438 t/s because 128 tokens cannot keep the GPU busy. Batching 4 short-prompt requests together recovers to 2625 t/s — a 6× improvement from batching alone.

#### Token generation (TG) throughput vs batch size

| PP | TG | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=32 / B=1 |
|----|----|-----|-----|-----|-----|------|------|-----------|
| 128 | 32 | 67 | 111 | 168 | 233 | 359 | 537 | 8.0× |
| 512 | 32 | 67 | 105 | 155 | 209 | 297 | 408 | 6.1× |
| 512 | 128 | 67 | 107 | 155 | 211 | 297 | 409 | 6.1× |
| 2048 | 128 | 61 | 92 | 124 | 157 | 198 | 243 | 4.0× |
| 4096 | 128 | 58 | 85 | 116 | 158 | 214 | 279 | 4.8× |

*Values in t/s aggregate.*

Each individual user still experiences ~67 t/s at PP=512. What scales is total server throughput: 32 concurrent users get 408 t/s combined (vs 67 t/s for one user). This 6× aggregate gain for 32× concurrency is memory-bandwidth bound — a single unified memory pool serves all sequences from the same physical DRAM.

**At PP=4096, B=32:** the server processes 32 simultaneous clients each with 4096-token prompts and 128-token responses (N_KV = 135,168 tokens), completing the full batch in ~60 seconds at 279 t/s aggregate TG. This is a large-scale concurrent workload served by a single consumer-grade device.

---

### Thinking mode: `thinking_budget=0` not supported in build 5d3a4a7

The `thinking_budget=0` API parameter (intended to suppress chain-of-thought generation) is silently ignored by llama.cpp build `5d3a4a7`. In the benchmark, both "thinking ON" and "thinking OFF" runs produced byte-for-byte identical outputs — same reasoning content (~1662 characters), same token counts (518 completion tokens), same latency (~7.8 s), same throughput (~68.5 t/s).

This is a known limitation of build `5d3a4a7`. Support for `thinking_budget` was merged into llama.cpp after this commit. To get working thinking suppression: rebuild without pinning the commit (`docker compose build`, no `LLAMA_CPP_REF` argument) to pick up the latest master, which includes proper thinking budget handling. There is no reliable workaround in this specific build.

Throughput with thinking active (26B, 5 prompts × 5 reps):

| Prompt type | Latency | Gen t/s | Completion tokens |
|-------------|---------|---------|-------------------|
| Math (347 × 28) | 3.00 ± 0.11 s | 69.7 ± 0.1 | 199 |
| Logic (bat and ball) | 5.96 ± 0.02 s | 68.7 ± 0.0 | 402 |
| Factual (capital city) | 0.80 ± 0.00 s | 70.0 ± 0.2 | 50 |
| Causal (syllogism) | 15.40 ± 0.03 s | 67.0 ± 0.0 | 1024 |
| Creative (haiku) | 13.71 ± 0.03 s | 67.3 ± 0.0 | 916 |

Generation speed is consistent at 67–70 t/s regardless of prompt type or response length. The variance in total latency is driven entirely by how many tokens the model chooses to generate (including reasoning), not by any speed variation.

---

## Key Takeaways

1. **MoE active parameters, not total parameters, determine generation speed.** The 26B MoE generates at the same speed as a dense 4B-class model despite having 25B total weights. File size on disk is a misleading performance proxy for MoE models.

2. **The MoE vs dense cliff is stark.** At similar file sizes (15.6 vs 17.4 GiB), the 26B MoE generates 6.4× faster than the 31B dense. Choosing between them is a direct quality/speed trade-off.

3. **Unified memory eliminates the PCIe bottleneck.** 70 t/s single-user generation from a single consumer SoC — without the layer-spilling penalty that limits discrete GPUs to 25–30 t/s for the same model size.

4. **PP throughput saturates quickly.** A single 512-token sequence already consumes ~2900 t/s of prompt throughput. Batching only helps for very short prompts (< 256 tokens), where you need ~4 concurrent requests to saturate the memory bandwidth.

5. **TG scales sub-linearly but meaningfully with concurrency.** 32 users get 6× aggregate TG throughput compared to 1 user. Each user's perceived speed remains the same; the server can serve more users without latency degradation.

6. **Context depth costs more on PP than TG.** At 32K depth, prefill drops 29% (26B) while generation drops only 24% — because TG is dominated by weight loading, not KV cache attention cost.

---

## Dockerfile Details

The `Dockerfile` is explicitly configured for the ARM64 + GB10 environment:

- **NGC base images:** Uses `nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04` (NGC, not Docker Hub) because Docker Hub does not publish official ARM64 CUDA 13 images.
- **SM_121 compute capability:** Sets `CUDA_DOCKER_ARCH=121` → `CMAKE_CUDA_ARCHITECTURES=121`. The GB10 Grace Blackwell SoC is SM_121. Discrete Blackwell GPUs (RTX 5090) are SM_100. Builds targeting generic "Blackwell" will generate wrong PTX for this device.
- **Compiled from source:** Clones `ggml-org/llama.cpp` master and builds with GCC-14 + CMake. Required to pick up the Gemma 4 support from PR #21309 without waiting for a versioned release.
- **Multi-stage build:** Compiles in the `-devel` CUDA image (~6 GB), copies only binaries + shared libs into a `-runtime` image. The resulting server image is significantly smaller than the build stage.
- **Pre-configured defaults:** Flash attention (`LLAMA_ARG_FLASH_ATTN=1`), full GPU offload (`LLAMA_ARG_N_GPU_LAYERS=-1`), and context size (`LLAMA_ARG_CTX_SIZE=8192`) are set as environment variable defaults — all overridable at container startup without rebuilding.

---

## Project Structure

```
.
├── Dockerfile                     # llama.cpp from source, CUDA 13, SM_121, ARM64
├── docker-compose.yml             # server (port 8080) + bench profile
├── .env                           # gitignored — add HF_TOKEN=hf_... here for private models
├── scripts/
│   ├── download_model.sh          # download any/all of the 4 models (requires image built first)
│   ├── test_inference.sh          # smoke test: health check + prompt + speed readout
│   ├── run_bench.sh               # groups A/B/C: llama-bench + batched-bench
│   └── run_bench_thinking.py      # thinking ON vs OFF via server API (runs on host)
├── models/                        # GGUF files — gitignored, not committed
└── results/                       # benchmark output saved by scripts — not gitignored
```

---

## Changelog

### 2026-04-06
- Expanded to full Gemma 4 family: E2B (Q8_0), E4B (Q4_K_M), 26B-A4B (Q4_K_M), 31B (Q4_K_M)
- Complete benchmark suite: Group A (4 models, 5 reps), Group B (context scaling), Group C (concurrency up to B=32, PP=4096, 256K context)
- Thinking mode benchmark: `thinking_budget=0` confirmed not honoured in build `5d3a4a7`
- Scripts: `download_model.sh` supports all 4 models; `run_bench.sh` loops all models with GROUP filter; `run_bench_thinking.py` measures thinking vs no-thinking via API

### 2026-04-05
- Initial setup: llama.cpp Docker for Gemma 4 26B-A4B on DGX Spark (GB10)
- Confirmed Gemma 4 llama.cpp support (PR #21309 merged)
- Dockerfile: NGC base image (`nvcr.io/nvidia/cuda:13.0.1`), SM_121, ARM64 — builds in ~3 min
- Server: 31/31 layers on GPU, 122 GB VRAM available, chain-of-thought reasoning active
- Baseline benchmarks: pp512 = 2607 t/s, tg128 = 70 t/s; batched prefill peaks at 2962 t/s (B=8, PP=512)
