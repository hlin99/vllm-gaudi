#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# Library paths needed for UCX/habanalabs (same as start_etcd_mooncake_master.sh)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/libfabric/lib:/usr/lib/habanalabs/:/usr/local/lib/
export LIBFABRIC_ROOT=/opt/libfabric-1.22.0
export LD_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib:${LD_LIBRARY_PATH}

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 0-7, IP 192.168.100.{231..238}
bash "$BASH_DIR"/../gen_decoder_configs.sh 231 0 8 "$BASH_DIR/.."

bash "$BASH_DIR"/../disaggregated_prefill_server_launcher.sh \
  -r decode \
  -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
  -n 8 -t 1 \
  --node-ip 192.168.100.231 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --kv-connector lmcache \
  --no-ep \
  --nixl-buffer-device hpu
