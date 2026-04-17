#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 8-15, IP 192.168.100.{221..228}
bash "$BASH_DIR"/../gen_decoder_configs.sh 221 1 8 "$BASH_DIR/.."

bash "$BASH_DIR"/../disaggregated_prefill_server_launcher.sh \
  -r decode \
  -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
  -n 8 -t 1 \
  --node-ip 192.168.100.221 \
  --node-rank 1 \
  --node-size 2 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache
