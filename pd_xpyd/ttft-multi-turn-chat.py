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

    # Load local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"{CYAN}--- PD Disaggregation Proxy Compatibility Mode (Token ID Sync) ---{RESET}")
    print(f"{CYAN}Log File: {log_filename}{RESET}\n")

    with open(log_filename, "a", encoding="utf-8") as log_file:
        while True:
            try:
                user_input = input(f"{GREEN}Boss >> {RESET}")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue

                is_first_turn = (len(current_context_ids) == 0)

                # Encode new turn text. Add BOS only if first message.
                new_turn_text = f"User: {user_input}\nAssistant: "
                new_ids = tokenizer.encode(new_turn_text, add_special_tokens=False)

                # Construct the full ID sequence
                send_ids = current_context_ids + new_ids
                print(f"DEBUG: current_context_ids len = {len(current_context_ids)}, "
                      f"new_ids len = {len(new_ids)}, send_ids len = {len(send_ids)}")

                # Reconstruct string from IDs
                current_full_prompt = tokenizer.decode(send_ids, skip_special_tokens=False)

                payload = {
                    "model": model_path,
                    "prompt": send_ids,
                    "max_tokens": 4000,
                    "temperature": 0,
                    "stream": True,
                    "add_special_tokens": False,
                    "stop": ["User:", "<｜end_of_sentence｜>"]
                }
                log_file.write(f"# SENT_PROMPT_LEN_TOKENS: {len(send_ids)}\n")

                # --- TTFT Timing ---
                ttf_start = time.time()
                response = requests.post(url, headers=headers, json=payload, stream=True, timeout=600)
                response.raise_for_status()

                print(f"{CYAN}Assistant >> {RESET}", end="", flush=True)

                this_turn_gen_ids = []
                latest_prompt_ids = []
                first_token_received = False

                # Stream processing
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode('utf-8')
                    log_file.write(line_str + "\n")
                    log_file.flush()

                    if line_str.startswith("data: "):
                        data_payload = line_str[6:].strip()
                        if data_payload == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_payload)
                            choice = chunk_json['choices'][0]

                            # Capture generated token IDs
                            t_ids = choice.get('token_ids', [])
                            if t_ids:
                                if not first_token_received:
                                    ttf_duration = time.time() - ttf_start
                                    print(f"\n[TTFT]: {ttf_duration:.3f} sec")
                                    first_token_received = True
                                this_turn_gen_ids.extend(t_ids)
                                sys.stdout.write(choice.get('text', ''))
                                sys.stdout.flush()

                            # Capture server-confirmed prompt IDs
                            p_ids = choice.get('prompt_token_ids', [])
                            if p_ids:
                                latest_prompt_ids = p_ids
                        except Exception:
                            # Ignore any malformed lines
                            continue

                # Synchronize context state for the next turn
                if latest_prompt_ids:
                    current_context_ids = latest_prompt_ids + this_turn_gen_ids
                else:
                    current_context_ids = send_ids + this_turn_gen_ids

                print("\n")

            except Exception as e:
                print(f"\n{GREEN}[Runtime Error]:{RESET} {e}")

if __name__ == "__main__":
    chat()

