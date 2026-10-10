#!/bin/bash
# send_request.sh - 向推理服务发送测试请求
#
# 用法:
#   scripts/send_request.sh [port] [host] [model] [--stream]
#   scripts/send_request.sh 8200
#   scripts/send_request.sh 8200 33.213.200.212 auto
#   scripts/send_request.sh 7001 localhost auto --stream

set -euo pipefail

PORT="${1:-8000}"
HOST="${2:-localhost}"
MODEL="${3:-auto}"
STREAM="${4:-}"

echo "=== Sending test request to ${HOST}:${PORT} ==="

if [[ "$STREAM" == "--stream" ]]; then
    echo "--- Stream mode ---"
    curl -sN "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you help me?"}
            ],
            "max_tokens": 100,
            "temperature": 0.6,
            "stream": true
        }'
    echo ""
else
    curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"}
            ],
            "ignore_eos": false,
            "stream": false,
            "temperature": 0.6,
            "top_k": 1,
            "max_tokens": 100
        }' | python3 -m json.tool 2>/dev/null || echo "Request failed or response is not valid JSON"
fi