import requests
import json
import sys

# 颜色定义
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

def chat():
    url = "http://10.239.129.9:8868/v1/completions"
    model_path = "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
    headers = {"Content-Type": "application/json"}
    
    # 核心：使用列表存储每一片段，最后精确拼接
    # 避免在每一轮中间手动添加不可控的 \n
    context_segments = []
    
    print(f"{CYAN}--- Prefix Caching 优化版 (Port: 8868) ---{RESET}")

    while True:
        try:
            user_input = input(f"\n{GREEN}海哥 >> {RESET}")
            if user_input.lower() in ['exit', 'quit']: break

            # 严格构造当前 Turn
            # 只有第一轮不带前缀换行，后续轮次保持结构严谨
            new_user_turn = f"User: {user_input}\nAssistant: "
            
            # 组合全量 Prompt
            current_prompt = "".join(context_segments) + new_user_turn

            payload = {
                "model": model_path,
                "prompt": current_prompt,
                "max_tokens": 2048,
                "temperature": 0, # 必须为 0 才能保证 Prefix Caching 收益最大化
                "stream": True,
                "stop": ["User:"] 
            }

            response = requests.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            print(f"{CYAN}Assistant >> {RESET}", end="", flush=True)
            
            this_turn_reply = ""
            for line in response.iter_lines():
                if not line: continue
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    content = line_str[6:]
                    if content == "[DONE]": break
                    try:
                        resp_json = json.loads(content)
                        delta = resp_json['choices'][0].get('text', '')
                        if delta:
                            print(delta, end="", flush=True)
                            this_turn_reply += delta
                    except: continue

            # 关键：精确存储，不额外加换行，除非模型自己输出了换行
            context_segments.append(new_user_turn)
            context_segments.append(this_turn_reply)
            # 只有在 Assistant 回复完后，强制加一个固定的换行符，确保下一次 User 开始时对齐
            context_segments.append("\n")

        except Exception as e:
            print(f"\n{GREEN}[错误]:{RESET} {e}")

if __name__ == "__main__":
    chat()
