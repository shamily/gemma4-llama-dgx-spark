#!/usr/bin/env bash
# Benchmark gemma-4-26B-A4B-it with llama-bench and llama-batched-bench.
# Mirrors the DGX Spark benchmark methodology from:
#   https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md
#
# Usage:
#   ./scripts/run_bench.sh
#   MODEL=./models/gemma-4-26b-a4b-it-Q8_0.gguf ./scripts/run_bench.sh

set -euo pipefail

MODEL="${MODEL:-./models/gemma-4-26B-A4B.MXFP4_MOE.gguf}"
RESULTS_DIR="./results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${RESULTS_DIR}/bench_${TIMESTAMP}.md"

mkdir -p "${RESULTS_DIR}"

GPU_INFO="$(docker run --rm --gpus all --entrypoint nvidia-smi llama_host:latest 2>/dev/null || echo 'unavailable')"

echo "# Benchmark Results — $(date -u '+%Y-%m-%d %H:%M UTC')" | tee "${OUT_FILE}"
echo "" | tee -a "${OUT_FILE}"
echo "**Model:** ${MODEL}" | tee -a "${OUT_FILE}"
echo "" | tee -a "${OUT_FILE}"
echo "## GPU Info" | tee -a "${OUT_FILE}"
echo "\`\`\`" | tee -a "${OUT_FILE}"
echo "${GPU_INFO}" | tee -a "${OUT_FILE}"
echo "\`\`\`" | tee -a "${OUT_FILE}"
echo "" | tee -a "${OUT_FILE}"

echo "## llama-bench (throughput)" | tee -a "${OUT_FILE}"
echo "\`\`\`" | tee -a "${OUT_FILE}"

docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-bench \
    llama_host:latest \
    --model "/models/$(basename "${MODEL}")" \
    --n-gpu-layers -1 \
    --flash-attn 1 \
    --n-ubatch 2048 \
    --output markdown \
    2>&1 | tee -a "${OUT_FILE}"

echo "\`\`\`" | tee -a "${OUT_FILE}"
echo "" | tee -a "${OUT_FILE}"

echo "## llama-batched-bench (multi-sequence throughput)" | tee -a "${OUT_FILE}"
echo "\`\`\`" | tee -a "${OUT_FILE}"

docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-batched-bench \
    llama_host:latest \
    --model "/models/$(basename "${MODEL}")" \
    --n-gpu-layers -1 \
    --ctx-size 270336 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --flash-attn \
    2>&1 | tee -a "${OUT_FILE}"

echo "\`\`\`" | tee -a "${OUT_FILE}"

echo ""
echo "==> Results saved to ${OUT_FILE}"
