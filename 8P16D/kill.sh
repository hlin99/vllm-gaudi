#!/bin/bash

set -euo pipefail

echo "[INFO] Running hl-smi and extracting PIDs..."

PIDS=$(hl-smi | awk '
/^|[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[A-Z]/ {
    pid=$3
    if (pid ~ /^[0-9]+$/) {
        print pid
    }
}' | sort -u)

if [ -z "$PIDS" ]; then
    echo "[INFO] No running compute processes found."
    exit 0
fi

echo "[INFO] Found the following PIDs:"
echo "$PIDS"

echo "[INFO] Killing processes..."
for pid in $PIDS; do
    echo "  kill -9 $pid"
    kill -9 "$pid" 2>/dev/null || echo "  [WARN] Failed to kill $pid"
done

echo "[INFO] Done."

