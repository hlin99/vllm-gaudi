#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export UCX_MEMTYPE_CACHE=0

# Generate prefiller configs with proxy internal IP
bash "$BASH_DIR/gen_prefiller_configs.sh" 192.168.100.191 1 8 "$BASH_DIR"

source "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
  -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
  -n 8 -t 1 \
  --node-ip 0.0.0.0 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache \
  --no-ep \
  --nixl-buffer-device hpu
