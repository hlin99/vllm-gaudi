#!/bin/bash
echo "bac"

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
echo "abc"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libfabric/lib:/usr/lib/habanalabs/:/usr/local/lib/
export REQUIRED_VERSION=1.22.0
export LIBFABRIC_ROOT=/opt/libfabric-1.22.0
export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MC_MS_AUTO_DISC=0

pkill -f mooncake_master
pkill -f etcd
sleep 5s

echo "aaa"

# Define commands as arrays
ETCD_CMD=(etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379)
MOON_CMD=(mooncake_master -rpc_thread_num 64 -rpc_port 50001 -eviction_high_watermark_ratio 0.8 -eviction_ratio 0.2 --v=1)

XPYD_LOG="${WORKSPACE_ROOT:-/workspace}/xpyd_logs"
echo "XPYD_LOG set to: $XPYD_LOG"
mkdir -p "$XPYD_LOG"

# Check if XPYD_LOG is set
if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")

    # Run etcd with logging
    ETCD_LOG="$XPYD_LOG/etcd_${timestamp}.log"
    echo "Starting etcd, logging to $ETCD_LOG..."
    "${ETCD_CMD[@]}" > "$ETCD_LOG" 2>&1 &

    # Run mooncake_master with logging
    MOON_LOG="$XPYD_LOG/mooncake_master_${timestamp}.log"
    echo "Starting mooncake_master, logging to $MOON_LOG..."
    "${MOON_CMD[@]}" > "$MOON_LOG" 2>&1 &
else
    # Run without logging
    echo "XPYD_LOG not set, running without logging..."
    "${ETCD_CMD[@]}" > /dev/null 2>&1 &
    "${MOON_CMD[@]}" > /dev/null 2>&1 &
fi
