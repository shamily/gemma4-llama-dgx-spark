#!/usr/bin/env bash
# Quick smoke test for the llama-server.
# Sends a single chat completion request and prints the response.
#
# Usage:
#   ./scripts/test_inference.sh
#   HOST=192.168.1.10 PORT=8080 ./scripts/test_inference.sh

set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"

echo "==> Waiting for server at ${BASE_URL}/health ..."
for i in $(seq 1 60); do
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "    Server is up."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Server did not become healthy after 60s."
        exit 1
    fi
    sleep 2
done

echo ""
echo "==> Server info:"
curl -s "${BASE_URL}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('data', []):
    print(f\"  model id : {m['id']}\")
    print(f\"  created  : {m.get('created')}\")
" 2>/dev/null || curl -s "${BASE_URL}/v1/models"

echo ""
echo "==> Sending test prompt..."
RESPONSE=$(curl -sf "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? Answer in one sentence."}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    }')

echo ""
echo "==> Response:"
echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
msg = data['choices'][0]['message']
usage = data.get('usage', {})
timings = data.get('timings', {})
reasoning = msg.get('reasoning_content', '')
if reasoning:
    print(f'  [thinking] {reasoning}')
    print()
print(f\"  {msg.get('content', '')}\")
print()
print(f\"  prompt tokens : {usage.get('prompt_tokens', '?')}\")
print(f\"  output tokens : {usage.get('completion_tokens', '?')}\")
print(f\"  prompt speed  : {timings.get('prompt_per_second', '?'):.1f} t/s\")
print(f\"  gen speed     : {timings.get('predicted_per_second', '?'):.1f} t/s\")
"

echo ""
echo "==> Test passed."
