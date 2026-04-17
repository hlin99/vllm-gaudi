#!/bin/bash
# Usage: bash gen_prefiller_configs.sh <proxy_internal_ip> <node_rank> <num_instances> [output_dir]
# Example: bash gen_prefiller_configs.sh 192.168.100.191 0 8 .

PROXY_IP=$1
NODE_RANK=$2
NUM_INSTANCES=$3
OUTPUT_DIR=${4:-.}
TEMPLATE="${OUTPUT_DIR}/lmcache-prefiller-config-template.yaml"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    GLOBAL_ID=$(( NODE_RANK * 8 + i ))
    OUTPUT="${OUTPUT_DIR}/lmcache-prefiller-config${GLOBAL_ID}.yaml"
    sed "s/__PD_PROXY_HOST__/${PROXY_IP}/" "$TEMPLATE" > "$OUTPUT"
    echo "Generated ${OUTPUT} -> pd_proxy_host=${PROXY_IP}"
done
