#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

source "$BASH_DIR/start_etcd_mooncake_master.sh"

export UCX_MEMTYPE_CACHE=0

# Generate prefiller configs with proxy internal IP
bash "$BASH_DIR/gen_prefiller_configs.sh" 192.168.100.191 0 8 "$BASH_DIR"

BASE_HTTP_PORT=8300

MLX5_DEVICES=(mlx5_0 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7 mlx5_8 mlx5_9)
MLX5_DEVICES=(mlx5_0 mlx5_0 mlx5_0 mlx5_0 mlx5_6 mlx5_7 mlx5_8 mlx5_9)

for i in $(seq 0 3); do
    GLOBAL_ID=$i
    PREFILL_IP="192.168.100.$((191 + i))"

    echo "====== Starting prefiller instance $i (IP: ${PREFILL_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    UCX_NET_DEVICES="${MLX5_DEVICES[$i]}:1" \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-prefiller-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
      -n 1 -t 1 \
      --node-ip "$PREFILL_IP" \
      --base-port "$((BASE_HTTP_PORT + i))" \
      --node-rank 0 \
      --node-size 1 \
      --max-model-len 16384 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.6 \
      --no-ep \
      --warmup \
      --nixl-buffer-device hpu \
      --recipe-cache \
      --kv-connector lmcache &

    sleep 2
done

echo "All 8 prefiller instances (machine1) launched. Waiting..."
wait
