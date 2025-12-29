#!/bin/bash

# Model path configuration
MODEL_PATH="/mnt/disk2/hf_models/DeepSeek-R1-G2/"

# Dataset configuration
DATASET="sonnet"
DATASET_PATH="benchmarks/sonnet.txt"

# Server connection details
HOST="10.239.129.9"
PORT="8868"

# List of input lengths to test
INPUT_LENS=(2000 5000 8000)

# List of concurrency levels to test
MAX_CONCURRENCY=(1 2 4 8 16)
# Iterate through defined input lengths
for input_len in "${INPUT_LENS[@]}"; do
    # Iterate through defined concurrency levels
    for concurrency in "${MAX_CONCURRENCY[@]}"; do

        # Dynamically set num_prompts based on concurrency and input length
        if [ "$concurrency" -eq 1 ]; then
            NUM_PROMPTS=64
        elif [ "$concurrency" -eq 2 ]; then
            NUM_PROMPTS=64
        else
            NUM_PROMPTS=128
        fi

        echo "Running benchmark with input_len=$input_len, max_concurrency=$concurrency, num_prompts=$NUM_PROMPTS"

        # Execute the benchmark script
        python3 benchmarks/benchmark_serving.py \
            --backend vllm \
            --model "$MODEL_PATH" \
            --dataset-name "$DATASET" \
            --request-rate inf \
            --host "$HOST" \
            --port "$PORT" \
            --sonnet-input-len "$input_len" \
            --sonnet-output-len 1 \
            --sonnet-prefix-len 100 \
            --trust-remote-code \
            --num-prompts "$NUM_PROMPTS" \
            --ignore-eos \
            --burstiness 1000 \
            --dataset-path "$DATASET_PATH" \
            --save-result \
            --max-concurrency "$concurrency"
    done
done

