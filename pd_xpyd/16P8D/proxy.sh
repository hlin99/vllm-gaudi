#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

python "$BASH_DIR"/../lmcache_proxy_server.py \
  --port 8868 \
  -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
  -p 192.168.100.191:8300-8307 192.168.100.211:8308-8315 \
  -d 192.168.100.221:9300-9307 \
  --bypass-proxy
