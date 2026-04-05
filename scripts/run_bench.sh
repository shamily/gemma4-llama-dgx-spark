#!/usr/bin/env bash
# Benchmark gemma-4-26B-A4B-it with llama-bench and llama-batched-bench.
# Mirrors the DGX Spark benchmark methodology from:
#   https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md
#
# Usage:
#   ./scripts/run_bench.sh
#   MODEL=./models/gemma-4-26b-a4b-it-Q8_0.gguf ./scripts/run_bench.sh

set -euo pipefail

MODEL="${MODEL:-./models/gemma-4-26B-A4B-it-Q4_K_M.gguf}"
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
    -m "/models/$(basename "${MODEL}")" \
    -ngl 99 \
    -fa 1 \
    -b 2048 \
    -ub 2048 \
    -o md \
    2>&1 | tee -a "${OUT_FILE}"

echo "\`\`\`" | tee -a "${OUT_FILE}"
echo "" | tee -a "${OUT_FILE}"

echo "## llama-batched-bench (multi-sequence throughput)" | tee -a "${OUT_FILE}"
echo "\`\`\`" | tee -a "${OUT_FILE}"

docker run --rm --gpus all \
    -v "$(pwd)/models:/models:ro" \
    --entrypoint /app/llama-batched-bench \
    llama_host:latest \
    -m "/models/$(basename "${MODEL}")" \
    -ngl 99 \
    -c 16384 \
    -b 2048 \
    -ub 2048 \
    -fa 1 \
    -npp 128,512,2048 \
    -ntg 32,128 \
    -npl 1,4,8 \
    2>&1 | tee -a "${OUT_FILE}"

echo "\`\`\`" | tee -a "${OUT_FILE}"

echo ""
echo "==> Results saved to ${OUT_FILE}"
