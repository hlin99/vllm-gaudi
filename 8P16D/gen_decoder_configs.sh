#!/bin/bash
# Usage: bash gen_decoder_configs.sh <base_ip_last_octet> <node_rank> <num_instances> [output_dir]
# Example: bash gen_decoder_configs.sh 221 0 8 pd_xpyd
#   Generates lmcache-decoder-config{0..7}.yaml with pd_peer_host 192.168.100.{221..228}

BASE_OCTET=$1
NODE_RANK=$2
NUM_INSTANCES=$3
OUTPUT_DIR=${4:-pd_xpyd}
TEMPLATE="${OUTPUT_DIR}/lmcache-decoder-config-template.yaml"

IP_PREFIX="192.168.100"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    GLOBAL_ID=$(( NODE_RANK * 8 + i ))
    PEER_IP="${IP_PREFIX}.$(( BASE_OCTET + i ))"
    OUTPUT="${OUTPUT_DIR}/lmcache-decoder-config${GLOBAL_ID}.yaml"

    sed "s/__PD_PEER_HOST__/${PEER_IP}/" "$TEMPLATE" > "$OUTPUT"
    echo "Generated ${OUTPUT} -> pd_peer_host=${PEER_IP}"
done
