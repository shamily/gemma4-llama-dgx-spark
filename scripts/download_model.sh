#!/usr/bin/env bash
# Download gemma-4-26B-A4B-it Q4_K_M GGUF from ggml-org (llama.cpp team).
# Public repo — no HF token required.
# Source: https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF
#
# Usage:
#   ./scripts/download_model.sh

set -euo pipefail

REPO="ggml-org/gemma-4-26B-A4B-it-GGUF"
MODELS_DIR="./models"

mkdir -p "${MODELS_DIR}"

echo "==> Downloading GGUF model files from ${REPO}..."

docker run --rm \
    -v "$(pwd)/models:/models" \
    --entrypoint python3 \
    llama_host:latest \
    -c "
from huggingface_hub import hf_hub_download
import os
repo = '${REPO}'
files = ['gemma-4-26B-A4B-it-Q4_K_M.gguf']
token = os.environ.get('HF_TOKEN')
for f in files:
    print(f'  -> {f}')
    hf_hub_download(repo_id=repo, filename=f, local_dir='/models', token=token)
print('Done.')
"

echo ""
echo "==> Done. Files:"
ls -lh "${MODELS_DIR}"/*.gguf 2>/dev/null || true
echo ""
echo "    Start the server with: docker compose up server"
