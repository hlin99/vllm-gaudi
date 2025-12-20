echo "--node-ip is $1"
echo "--dp-master-ip is $2"
echo "--node-rank is $3"

bash disaggregated_prefill_server_launcher.sh -n 8 -d 16 --node-ip $1 --node-size 2 --node-rank $3 --dp-master-ip $2 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 32 -r decode --async --nixl-buffer-device hpu --log-dir /workspace $4
#bash /mnt/disk3/xinyu/ml.infra.tools/scripts/nixl_connector/disaggregated_prefill_server_launcher -m /mnt/disk2/hf_models/DeepSeek-R1-G2-static/ --node-size 2 --node-rank 0 --dp-master-ip 10.239.129.81 -d 16 -n 8 --node-ip 192.168.100.221 --max-model-len 16384 --max-num-seqs 64 -r decode --async --nixl-buffer-device hpu
#bash disaggregated_prefill_server_launcher.sh -m /mnt/disk2/hf_models/DeepSeek-R1-G2-static/ --node-size 2 --node-rank 0 --dp-master-ip 10.239.129.81 -d 16 -n 8 --node-ip 192.168.100.221 --max-model-len 16384 --max-num-seqs 64 -r decode --async --nixl-buffer-device hpu

