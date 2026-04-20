#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export UCX_MEMTYPE_CACHE=0

# Generate prefiller configs with proxy internal IP
bash "$BASH_DIR/gen_prefiller_configs.sh" 192.168.100.191 1 8 "$BASH_DIR"

BASE_HTTP_PORT=8300

for i in $(seq 0 7); do
    GLOBAL_ID=$((8 + i))
    PREFILL_IP="192.168.100.$((221 + i))"

    echo "====== Starting prefiller instance $i (IP: ${PREFILL_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-prefiller-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
      -n 1 -t 1 \
      --node-ip "$PREFILL_IP" \
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

echo "All 8 prefiller instances (machine2) launched. Waiting..."
wait
