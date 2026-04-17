#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 0-7, IP 192.168.100.{211..218}
bash "$BASH_DIR/gen_decoder_configs.sh" 211 0 8 "$BASH_DIR"

bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
  -r decode \
  -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
  -n 8 -t 1 \
  --node-ip 192.168.100.211 \
  --node-rank 0 \
  --node-size 2 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache
