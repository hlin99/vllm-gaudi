BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# Normal Mode
python "$BASH_DIR"/proxy_server.py --port 8868 -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -p 192.168.100.191:8300 -d 192.168.100.221:9300-9307 192.168.100.231:9308-9315 --bypass-proxy

# Benchmark Mode
# python "$BASH_DIR"/proxy_server.py --port 8868 -m /mnt/disk2/hf_models/DeepSeek-R1-G2/ -p 192.168.100.191:8300 -d 192.168.100.221:9300-9307 192.168.100.231:9308-9315 --bypass-proxy --repeat_p_request 1 --repeat_d_times 511 --benchmark_mode
