#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python "$BASH_DIR/proxy_server.py" \
  --port 8868 \
  -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
  --bypass-proxy \
  -p 192.168.100.191:8300 \
  -d 192.168.100.221:9300 \
  --decoder-init-port 7300 \
  --decoder-alloc-port 7400 \
  --proxy-host 0.0.0.0 \
  --proxy-port 7500 \
  --lmcache_nixl

