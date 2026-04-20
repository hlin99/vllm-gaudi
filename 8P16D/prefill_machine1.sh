#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

source "$BASH_DIR/start_etcd_mooncake_master.sh"

export UCX_MEMTYPE_CACHE=0

# Generate prefiller configs with proxy internal IP
bash "$BASH_DIR/gen_prefiller_configs.sh" 192.168.100.191 0 8 "$BASH_DIR"

BASE_HTTP_PORT=8300

for i in $(seq 0 7); do
    GLOBAL_ID=$i
    PREFILL_IP="192.168.100.$((191 + i))"

    echo "====== Starting prefiller instance $i (IP: ${PREFILL_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-prefiller-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -m /mnt/disk2/hf_models/Meta-Llama-3-8B-Instruct/ \
      -n 1 -t 1 \
      --node-ip "$PREFILL_IP" \
      --base-port "$((BASE_HTTP_PORT + i))" \
      --node-rank 0 \
      --node-size 1 \
      --max-model-len 4096 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 128 \
      --gpu-memory-utilization 0.8 \
      --kv-connector lmcache &

    sleep 2
done

echo "All 8 prefiller instances (machine1) launched. Waiting..."
wait
