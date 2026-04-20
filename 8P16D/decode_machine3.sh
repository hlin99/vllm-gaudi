#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 8-15, IP 192.168.100.{221..228}, ports 7308-7315/7408-7415
bash "$BASH_DIR/gen_decoder_configs.sh" 221 1 8 "$BASH_DIR" 7308 7408

BASE_HTTP_PORT=8300

for i in $(seq 0 7); do
    GLOBAL_ID=$((8 + i))
    RDMA_IP="192.168.100.$((221 + i))"

    echo "====== Starting decoder instance $i (RDMA: ${RDMA_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-decoder-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -r decode \
      -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
      -n 1 -t 1 \
      --node-ip "$RDMA_IP" \
      --base-port "$((BASE_HTTP_PORT + i))" \
      --node-rank 1 \
      --node-size 2 \
      --max-model-len 4096 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 128 \
      --gpu-memory-utilization 0.8 \
      --no-ep \
      --kv-connector lmcache &

    sleep 2
done

echo "All 8 decoder instances (machine3) launched. Waiting..."
wait
