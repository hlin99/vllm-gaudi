BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

echo "--node-ip is $1"
echo "--dp-master-ip is $2"
echo "--node-rank is $3"

bash "$BASH_DIR"/disaggregated_prefill_server_launcher.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -n 8 -d 16 --node-ip $1 --node-size 2 --node-rank $3 --dp-master-ip $2 --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 -r decode --async --nixl-buffer-device hpu --inc /workspace/vllm-gaudi/pd_xpyd/inc_ep16/maxabs_quant_g2_ep16.json --log-dir /workspace --warmup --recipe-cache $4
