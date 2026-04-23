#!/bin/bash
BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export UCX_MEMTYPE_CACHE=0

# Generate decoder configs: global_id 0-7, IP 192.168.100.{221..228}, ports 7300-7307/7400-7407
bash "$BASH_DIR/gen_decoder_configs.sh" 221 0 8 "$BASH_DIR" 7300 7400

BASE_HTTP_PORT=8300

MLX5_DEVICES=(mlx5_0 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7 mlx5_8 mlx5_9)

for i in $(seq 0 0); do
    GLOBAL_ID=$i
    RDMA_IP="192.168.100.$((221 + i))"

    echo "====== Starting decoder instance $i (RDMA: ${RDMA_IP}, HTTP: $((BASE_HTTP_PORT + i))) ======"

    HABANA_VISIBLE_DEVICES=$i \
    UCX_NET_DEVICES="${MLX5_DEVICES[$i]}:1" \
    LMCACHE_CONFIG_FILE="${BASH_DIR}/lmcache-decoder-config${GLOBAL_ID}.yaml" \
    bash "$BASH_DIR/disaggregated_prefill_server_launcher.sh" \
      -r decode \
      -m /mnt/disk2/hf_models/DeepSeek-V2-Lite-Chat/ \
      -n 1 -t 1 \
      --node-ip "$RDMA_IP" \
      --base-port "$((BASE_HTTP_PORT + i))" \
      --node-rank 0 \
      --node-size 2 \
      --max-model-len 16384 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.6 \
      --no-ep \
      --warmup \
      --recipe-cache \
      --nixl-buffer-device hpu \
      --kv-connector lmcache &

    sleep 2
done

echo "All 8 decoder instances (machine2) launched. Waiting..."
wait
