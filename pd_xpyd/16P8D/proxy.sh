#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

python "$BASH_DIR"/../proxy_server.py \
  --port 8868 \
  -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
  -p 192.168.100.191:8300 \
  -d 192.168.100.221:9300 \
  --bypass-proxy \
