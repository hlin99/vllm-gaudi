#!/bin/bash
# Usage: bash gen_decoder_configs.sh <base_ip_last_octet> <node_rank> <num_instances> [output_dir] [base_init_port] [base_alloc_port]
# Example: bash gen_decoder_configs.sh 221 0 8 pd_xpyd 7300 7400
#   Generates lmcache-decoder-config{0..7}.yaml with pd_peer_host 192.168.100.{221..228}
#   and unique init/alloc ports 7300-7307 / 7400-7407

BASE_OCTET=$1
NODE_RANK=$2
NUM_INSTANCES=$3
OUTPUT_DIR=${4:-pd_xpyd}
BASE_INIT_PORT=${5:-7300}
BASE_ALLOC_PORT=${6:-7400}
TEMPLATE="${OUTPUT_DIR}/lmcache-decoder-config-template.yaml"

IP_PREFIX="192.168.100"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    GLOBAL_ID=$(( NODE_RANK * 8 + i ))
    PEER_IP="${IP_PREFIX}.$(( BASE_OCTET + i ))"
    INIT_PORT=$(( BASE_INIT_PORT + i ))
    ALLOC_PORT=$(( BASE_ALLOC_PORT + i ))
    OUTPUT="${OUTPUT_DIR}/lmcache-decoder-config${GLOBAL_ID}.yaml"

    sed -e "s/__PD_PEER_HOST__/${PEER_IP}/" \
        -e "s/__PD_PEER_INIT_PORT__/${INIT_PORT}/" \
        -e "s/__PD_PEER_ALLOC_PORT__/${ALLOC_PORT}/" \
        "$TEMPLATE" > "$OUTPUT"
    echo "Generated ${OUTPUT} -> pd_peer_host=${PEER_IP}, init_port=${INIT_PORT}, alloc_port=${ALLOC_PORT}"
done
