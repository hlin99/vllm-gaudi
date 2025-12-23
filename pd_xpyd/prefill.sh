BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# 16K model len for prefill
bash "$BASH_DIR"/disaggregated_prefill_server_launcher.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -n 1 -t 8 --node-ip 192.168.100.191 --max-model-len 4096 --max-num-batched-tokens 8192 --max-num-seqs 8 --nixl-buffer-device hpu --log-dir /workspace --gpu-memory-utilization 0.5 --inc /workspace/vllm-gaudi/pd_xpyd/inc_ep8/maxabs_quant_g2_ep8.json --recipe-cache --warmup

# 128K model len for prefill
# bash "$BASH_DIR"/disaggregated_prefill_server_launcher.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -n 1 -t 8 --node-ip 192.168.100.191 --max-model-len 131072 --max-num-batched-tokens 8192 --max-num-seqs 1 --nixl-buffer-device hpu --log-dir /workspace --gpu-memory-utilization 0.711 --inc /workspace/vllm-gaudi/pd_xpyd/inc_ep8/maxabs_quant_g2_ep8.json --apc --warmup


#bash /mnt/disk3/xinyu/ml.infra.tools/scripts/nixl_connector/disaggregated_prefill_server_launcher -m /mnt/disk2/hf_models/DeepSeek-R1-G2-static/ -n 1 -t 8 --node-ip 192.168.100.191 --max-model-len 16384 --max-num-seqs 1 --nixl-buffer-device hpu  --gpu-memory-utilization 0.7 --max-num-batched-tokens 16384
