#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Library paths needed for UCX/habanalabs (same as start_etcd_mooncake_master.sh)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/libfabric/lib:/usr/lib/habanalabs/:/usr/local/lib/
export LIBFABRIC_ROOT=/opt/libfabric-1.22.0
export LD_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib:${LD_LIBRARY_PATH}

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 0-7, IP 192.168.100.{231..238}, ports 7300-7307/7400-7407
bash "$BASH_DIR/gen_decoder_configs.sh" 231 0 8 "$BASH_DIR" 7300 7400

BASE_HTTP_PORT=9300

for i in $(seq 0 7); do
    GLOBAL_ID=$i
    RDMA_IP="192.168.100.$((231 + i))"

    echo "====== Starting decoder instance $i (RDMA: ${RDMA_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-decoder-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -r decode \
      -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
      -n 1 -t 1 \
      --node-ip "$RDMA_IP" \
      --base-port "$((BASE_HTTP_PORT + i))" \
      --max-model-len 4096 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 128 \
      --gpu-memory-utilization 0.8 \
      --kv-connector lmcache \
      --no-ep \
      --nixl-buffer-device hpu &

    sleep 2
done

echo "All 8 decoder instances launched. Waiting..."
wait
