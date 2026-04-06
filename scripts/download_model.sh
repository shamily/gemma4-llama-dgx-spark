#!/usr/bin/env bash
# Download Gemma 4 GGUF models from ggml-org (llama.cpp team).
# Public repos — no HF token required.
#
# Models:
#   e2b  gemma-4-E2B-it   Q8_0    4.97 GB   (no Q4_K_M available upstream)
#   e4b  gemma-4-E4B-it   Q4_K_M  5.34 GB
#   26b  gemma-4-26B-A4B  Q4_K_M  15.6 GB
#   31b  gemma-4-31B-it   Q4_K_M  18.7 GB
#
# Usage:
#   ./scripts/download_model.sh              # download all 4 models
#   ./scripts/download_model.sh e2b e4b      # download specific models

set -euo pipefail

MODELS_DIR="./models"
mkdir -p "${MODELS_DIR}"

# -- model registry ---------------------------------------------------------
# key -> "repo|filename"
declare -A REGISTRY=(
  [e2b]="ggml-org/gemma-4-E2B-it-GGUF|gemma-4-e2b-it-Q8_0.gguf"
  [e4b]="ggml-org/gemma-4-E4B-it-GGUF|gemma-4-e4b-it-Q4_K_M.gguf"
  [26b]="ggml-org/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-Q4_K_M.gguf"
  [31b]="ggml-org/gemma-4-31B-it-GGUF|gemma-4-31B-it-Q4_K_M.gguf"
)

if [[ $# -eq 0 ]]; then
  TARGETS=(e2b e4b 26b 31b)
else
  TARGETS=("$@")
fi

for key in "${TARGETS[@]}"; do
  if [[ -z "${REGISTRY[$key]+x}" ]]; then
    echo "ERROR: unknown model key '${key}'. Valid keys: ${!REGISTRY[*]}"
    exit 1
  fi

  IFS='|' read -r repo filename <<< "${REGISTRY[$key]}"
  dest="${MODELS_DIR}/${filename}"

  if [[ -f "${dest}" ]]; then
    size=$(du -sh "${dest}" | cut -f1)
    echo "==> ${filename} already exists (${size}), skipping."
    continue
  fi

  echo "==> Downloading ${filename} from ${repo} ..."
  docker run --rm \
    -v "$(pwd)/models:/models" \
    --entrypoint python3 \
    llama_host:latest \
    -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='${repo}',
    filename='${filename}',
    local_dir='/models',
    token=os.environ.get('HF_TOKEN'),
)
print('Done: ${filename}')
"
done

echo ""
echo "==> Models in ${MODELS_DIR}:"
ls -lh "${MODELS_DIR}"/*.gguf 2>/dev/null || echo "  (none)"
