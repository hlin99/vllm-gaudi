BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# 16K model len for prefill
bash "$BASH_DIR"/single_prefill.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -n 1 -t 8 --node-ip 192.168.100.191 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 8 --nixl-buffer-device hpu --log-dir /workspace --gpu-memory-utilization 0.5 --inc /workspace/vllm-gaudi/pd_xpyd/inc_ep8/maxabs_quant_g2_ep8.json

