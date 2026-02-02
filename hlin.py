import requests
import json
import sys
import time
from transformers import AutoTokenizer

# Terminal enhancement
try:
    import readline
except ImportError:
    pass

GREEN, CYAN, RESET = "\033[92m", "\033[96m", "\033[0m"

def chat():
    # Target proxy URL
    url = "http://10.239.129.9:8868/v1/completions"
    model_path = "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
    headers = {"Content-Type": "application/json"}

    # Initialize current context as a list of Token IDs
    current_context_ids = []
    log_filename = f"api_raw_log_{int(time.time())}.jsonl"

    # Load local tokenizer to handle ID-to-String reconstruction
    tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/hf_models/DeepSeek-R1-G2/")

    print(f"{CYAN}--- PD Disaggregation Proxy Compatibility Mode (Token ID Sync) ---{RESET}")
    print(f"{CYAN}Log File: {log_filename}{RESET}\n")

    with open(log_filename, "a", encoding="utf-8") as log_file:
        while True:
            try:
                user_input = input(f"{GREEN}海哥 >> {RESET}")
                if user_input.lower() in ['exit', 'quit']: break
                if not user_input.strip(): continue

                # 1. Encode new turn text. 
                # Add BOS (special token) only if it's the very first message.
                new_turn_text = f"User: {user_input}\nAssistant: "
                new_ids = tokenizer.encode(new_turn_text, add_special_tokens=False)
                
                # 2. Construct the logical full ID sequence
                send_ids = current_context_ids + new_ids
                
                # 3. Reconstruct string from IDs to bypass Proxy's "No List" restriction.
                # Using skip_special_tokens=False is vital to keep the BOS and structural tokens.
                current_full_prompt = tokenizer.decode(send_ids, skip_special_tokens=False)

                payload = {
                    "model": model_path,
                    "prompt": current_full_prompt,
                    "prompt_token_ids": send_ids,
                    "max_tokens": 4096,
                    "temperature": 0,  # Zero temp helps verify hash stability
                    "stream": True,
                    "add_special_tokens": False,
                    "stop": ["User:", "<｜end_of_sentence｜>"]
                }
                print("send_ids=", send_ids)
                import hashlib  # Add this line
                import numpy as np

                # 假设 prefix_tokens 是前 256 个 ID
                prefix_array = np.array(send_ids[:256], dtype=np.int32)

                # 注意：具体算法取决于后端实现（vLLM 常用的是对 ID 序列做逻辑合并再 Hash）
                # 简单的模拟方式如下：
                local_chunk_hash = hash(prefix_array.tobytes()) 
                print(f"本地模拟 Hash (十进制): {local_chunk_hash}")

                #prefix_tokens = send_ids[:256]
                # Convert list of ints to bytes for hashing
                #token_bytes = json.dumps(prefix_tokens).encode('utf-8')
                #prefix_hash = hashlib.md5(token_bytes).hexdigest()

                #print(" prefix_hash=", prefix_hash)
                # Record metadata for debugging hash alignment
                log_file.write(f"# SENT_PROMPT_LEN_TOKENS: {len(send_ids)}\n")

                response = requests.post(url, headers=headers, json=payload, stream=True, timeout=600)
                response.raise_for_status()

                print(f"{CYAN}Assistant >> {RESET}", end="", flush=True)

                this_turn_gen_ids = []
                latest_prompt_ids = []

                # 4. Stream processing: capture both generated tokens and server-side prompt tokens
                for line in response.iter_lines():
                    if not line: continue
                    line_str = line.decode('utf-8')
                    log_file.write(line_str + "\n")
                    log_file.flush()

                    if line_str.startswith("data: "):
                        data_payload = line_str[6:].strip()
                        if data_payload == "[DONE]": break
                        try:
                            chunk_json = json.loads(data_payload)
                            choice = chunk_json['choices'][0]
                            
                            # Capture generated token IDs (decode phase)
                            t_ids = choice.get('token_ids', [])
                            if t_ids:
                                this_turn_gen_ids.extend(t_ids)
                                sys.stdout.write(choice.get('text', ''))
                                sys.stdout.flush()

                            # Capture server-confirmed prompt IDs (prefill phase)
                            p_ids = choice.get('prompt_token_ids')
                            if p_ids:
                                latest_prompt_ids = p_ids
                                print(" latest_prompt_ids=", latest_prompt_ids)
                        except: continue

                # 5. Synchronize context state for the next turn
                # Prioritize server-returned prompt IDs as they represent the actual KV Cache state
                if latest_prompt_ids:
                    current_context_ids = latest_prompt_ids + this_turn_gen_ids
                else:
                    # Fallback to local reconstruction if server metadata is missing
                    current_context_ids = send_ids + this_turn_gen_ids

                print("\n")

            except Exception as e:
                print(f"\n{GREEN}[Runtime Error]:{RESET} {e}")

if __name__ == "__main__":
    chat()
