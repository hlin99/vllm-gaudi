#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python "$BASH_DIR/lmcache_proxy_server.py" \
  --port 8868 \
  -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
  -p 192.168.100.191:8300-8307 \
  -d 192.168.100.211:9300-9307 192.168.100.221:9308-9315 \
  --bypass-proxy
