#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

source pd_xpyd/start_etcd_mooncake_master.sh

export UCX_MEMTYPE_CACHE=0

source "$BASH_DIR"/../disaggregated_prefill_server_launcher.sh \
  -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
  -n 8 -t 1 \
  --node-ip 192.168.100.191 \
  --node-rank 0 \
  --node-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache
