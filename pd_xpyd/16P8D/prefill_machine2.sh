#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

export UCX_MEMTYPE_CACHE=0

source "$BASH_DIR"/../disaggregated_prefill_server_launcher.sh \
  -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
  -n 1 -t 1 \
  --node-ip 192.168.100.221 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache \
  --no-ep
