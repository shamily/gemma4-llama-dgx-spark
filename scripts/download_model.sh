#!/usr/bin/env bash
# Download gemma-4-26B-A4B MXFP4_MOE GGUF from HuggingFace.
# Public repo — no HF token required.
# Source: https://huggingface.co/tianrui6641/gemma-4-26b-a4b-gguf-mxfp4-moe
#
# Usage:
#   ./scripts/download_model.sh

set -euo pipefail

REPO="tianrui6641/gemma-4-26b-a4b-gguf-mxfp4-moe"
MODELS_DIR="./models"

mkdir -p "${MODELS_DIR}"

echo "==> Downloading GGUF model files from ${REPO} via Docker..."

docker run --rm \
    -v "$(pwd)/models:/models" \
    --entrypoint python3 \
    llama_host:latest \
    -c "
from huggingface_hub import hf_hub_download
import os
repo = '${REPO}'
files = ['gemma-4-26B-A4B.MXFP4_MOE.gguf', 'mmproj-gemma-4-26B-A4B.f16.gguf']
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
