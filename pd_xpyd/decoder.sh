BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

bash "$BASH_DIR"/disaggregated_prefill_server_launcher.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -n 1 -t 8 --node-ip 192.168.100.191 --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 8 --nixl-buffer-device hpu --log-dir /workspace --gpu-memory-utilization 0.35 -r decode --kv-connector lmcache --inc /workspace/vllm-gaudi/pd_xpyd/inc_ep8/maxabs_quant_g2_ep8.json
