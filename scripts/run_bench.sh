#!/usr/bin/env bash
# Comprehensive benchmark for all Gemma 4 models on llama.cpp.
# Mirrors the DGX Spark benchmark methodology from:
#   https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md
#
# Three groups:
#   A  llama-bench throughput      — all 4 models, 5 reps
#   B  llama-bench context scaling — 26b + 31b only, 3 reps, pp up to 32k
#   C  llama-batched-bench         — 26b only, concurrency scaling
#
# Usage:
#   ./scripts/run_bench.sh              # run all groups, all models
#   GROUP=A ./scripts/run_bench.sh      # run only group A
#   SKIP_MISSING=1 ./scripts/run_bench.sh  # skip models not yet downloaded

set -euo pipefail

RESULTS_DIR="./results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${RESULTS_DIR}/bench_${TIMESTAMP}.md"
GROUP="${GROUP:-ALL}"
SKIP_MISSING="${SKIP_MISSING:-0}"

mkdir -p "${RESULTS_DIR}"

# -- model registry ---------------------------------------------------------
# Display name -> filename
declare -A MODELS=(
  [gemma-4-E2B-it-Q8_0]="gemma-4-e2b-it-Q8_0.gguf"
  [gemma-4-E4B-it-Q4_K_M]="gemma-4-e4b-it-Q4_K_M.gguf"
  [gemma-4-26B-A4B-it-Q4_K_M]="gemma-4-26B-A4B-it-Q4_K_M.gguf"
  [gemma-4-31B-it-Q4_K_M]="gemma-4-31B-it-Q4_K_M.gguf"
)
# Ordered list for consistent output
MODEL_KEYS=(
  "gemma-4-E2B-it-Q8_0"
  "gemma-4-E4B-it-Q4_K_M"
  "gemma-4-26B-A4B-it-Q4_K_M"
  "gemma-4-31B-it-Q4_K_M"
)
# Models for context scaling (Group B)
CONTEXT_SCALE_KEYS=(
  "gemma-4-26B-A4B-it-Q4_K_M"
  "gemma-4-31B-it-Q4_K_M"
)
# Model for concurrency benchmark (Group C)
BATCHED_KEY="gemma-4-26B-A4B-it-Q4_K_M"

# -- helpers ----------------------------------------------------------------
log() { echo "$*" | tee -a "${OUT_FILE}"; }
log_raw() { echo "$*" >> "${OUT_FILE}"; }

check_model() {
  local key="$1"
  local file="./models/${MODELS[$key]}"
  if [[ ! -f "${file}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "==> SKIP: ${file} not found."
      return 1
    else
      echo "ERROR: ${file} not found. Run ./scripts/download_model.sh first."
      exit 1
    fi
  fi
  return 0
}

run_llama_bench() {
  local label="$1"; shift
  local model_file="$1"; shift
  # remaining args passed to llama-bench
  log ""
  log "### ${label}"
  log ""
  docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-bench \
    llama_host:latest \
    -m "/models/${model_file}" \
    -ngl 99 \
    -fa 1 \
    -b 2048 \
    -ub 2048 \
    -o md \
    "$@" \
    2>/dev/null | tee -a "${OUT_FILE}"
}

run_batched_bench() {
  local label="$1"; shift
  local model_file="$1"; shift
  log ""
  log "### ${label}"
  log ""
  docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-batched-bench \
    llama_host:latest \
    -m "/models/${model_file}" \
    -ngl 99 \
    -b 2048 \
    -ub 2048 \
    -fa 1 \
    "$@" \
    2>/dev/null | tee -a "${OUT_FILE}"
}

# -- header -----------------------------------------------------------------
{
  echo "# Gemma 4 Benchmark Results — $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo ""
  echo "## Environment"
  echo ""
  echo '```'
  echo "OS:        $(uname -srm)"
  docker run --rm --gpus all \
    nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 \
    bash -c "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | \
             awk -F',' '{printf \"GPU:       %s\\nSM:        %s\\nVRAM:      %s\\nDriver:    %s\\n\", \$1,\$2,\$3,\$4}'" \
    2>/dev/null || echo "GPU:       (nvidia-smi unavailable)"
  echo "CUDA:      13.0"
  docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-bench \
    llama_host:latest --version 2>&1 | grep -oP 'build: \S+' | head -1 | \
    sed 's/build:/llama.cpp:/' || echo "llama.cpp: (version unavailable)"
  echo '```'
  echo ""
} | tee "${OUT_FILE}"

# -- Group A: throughput, all models ----------------------------------------
if [[ "${GROUP}" == "ALL" || "${GROUP}" == "A" ]]; then
  log "## Group A — Single-sequence throughput (all models)"
  log ""
  log "Settings: \`llama-bench\`, flash attention on, batch 2048, **5 repetitions**"
  log ""
  log "Tests: pp512, pp2048, tg32, tg128"
  log ""

  for key in "${MODEL_KEYS[@]}"; do
    check_model "${key}" || continue
    run_llama_bench "${key}" "${MODELS[$key]}" \
      -p 512,2048 \
      -n 32,128 \
      -r 5
  done
fi

# -- Group B: context scaling, 26b + 31b ------------------------------------
if [[ "${GROUP}" == "ALL" || "${GROUP}" == "B" ]]; then
  log ""
  log "## Group B — Context window scaling (26B and 31B)"
  log ""
  log "Settings: \`llama-bench\`, **3 repetitions**"
  log ""
  log "Tests: pp512 + tg128 at KV-cache depths 0 → 32768 — shows how throughput degrades as context fills"
  log ""

  for key in "${CONTEXT_SCALE_KEYS[@]}"; do
    check_model "${key}" || continue
    run_llama_bench "${key} — context scaling" "${MODELS[$key]}" \
      -p 512 \
      -n 128 \
      -d 0,4096,8192,16384,32768 \
      -r 3
  done
fi

# -- Group C: concurrency scaling, 26b only ---------------------------------
if [[ "${GROUP}" == "ALL" || "${GROUP}" == "C" ]]; then
  log ""
  log "## Group C — Multi-sequence concurrency (26B)"
  log ""
  log "Settings: \`llama-batched-bench\`, context 65536, flash attention on, batch 2048"
  log ""
  log "Tests: PP ∈ {128, 512, 2048, 4096}, TG ∈ {32, 128}, B ∈ {1, 2, 4, 8, 16, 32}"
  log ""

  check_model "${BATCHED_KEY}" && \
    run_batched_bench "${BATCHED_KEY} — concurrency" "${MODELS[$BATCHED_KEY]}" \
      -npp 128,512,2048,4096 \
      -ntg 32,128 \
      -npl 1,2,4,8,16,32 \
      -c 262144
fi

echo ""
echo "==> Results saved to ${OUT_FILE}"
