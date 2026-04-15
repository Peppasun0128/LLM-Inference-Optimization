import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class FinalInferenceEngine:
    def __init__(self, model_name="/model/Llama-3.2-3B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[System] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[System] Loading Model (bfloat16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print("[System] Engine Ready.\n")

    def generate(self, prompt, max_len=200, temp=1.0, 
                 top_k=None, top_p=None, min_p=None, 
                 rep_p=1.0, freq_p=0.0, pres_p=0.0, greedy=False):
        
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)
        
        seen_counts = {}
        output_ids = input_ids.clone()

        for _ in range(max_len):
            with torch.no_grad():
                logits = self.model(output_ids).logits[:, -1, :]
            
            # --- Penalty 機制 ---
            for token_id, count in seen_counts.items():
                if count > 0:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= rep_p
                    else:
                        logits[0, token_id] *= rep_p
                    logits[0, token_id] -= pres_p
                logits[0, token_id] -= (count * freq_p)

            # --- Decoding & Sampling 機制 ---
            if greedy:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                temp = max(temp, 1e-5)
                logits = logits / temp
                probs = torch.softmax(logits, dim=-1)

                if top_k is not None:
                    top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_values[..., -1, None]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                if min_p is not None:
                    max_prob = torch.max(probs, dim=-1, keepdim=True).values
                    indices_to_remove = probs < (max_prob * min_p)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                final_probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(final_probs, num_samples=1)

            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            tid = next_token_id.item()
            seen_counts[tid] = seen_counts.get(tid, 0) + 1
            
            if tid == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def run_all_tasks():
    engine = FinalInferenceEngine()
    
    print("="*60)
    print("1. Temperature 推論測試 | 題目: 用一句話形容今天的心情")
    print("-" * 60)
    q1 = "請用一句話形容今天的心情。"
    print(f"【設定: 無 (Temp=1.0)】\n{engine.generate(q1, max_len=80, temp=1.0, top_p=0.95)}\n")
    print(f"【設定: Temperature (Temp=0.1)】\n{engine.generate(q1, max_len=80, temp=0.1)}\n")

    print("="*60)
    print("2. Sampling 推論測試 | 題目: 續寫故事")
    print("-" * 60)
    q2 = "請接續以下句子完成一句話的故事：“今天早上我打開冰箱，發現...”"
    print(f"【設定: 無 (加入輕微Top-K安全網避免亂碼)】\n{engine.generate(q2, max_len=150, temp=0.8, top_k=100)}\n")
    print(f"【設定: top-K (K=5)】\n{engine.generate(q2, max_len=150, temp=0.8, top_k=5)}\n")
    print(f"【設定: top-P (P=0.9)】\n{engine.generate(q2, max_len=150, temp=0.8, top_p=0.9)}\n")
    print(f"【設定: min-P (P=0.05)】\n{engine.generate(q2, max_len=150, temp=0.8, min_p=0.05)}\n")

    print("="*60)
    print("3. Decoding 推論測試 | 題目: 三種奇怪的食物組合")
    print("-" * 60)
    q3 = "請列出三種奇怪的食物組合，並簡短說明理由。"
    # 加大 max_len 確保三種食物都能寫完
    print(f"【設定: random (加入安全網)】\n{engine.generate(q3, max_len=350, temp=0.9, top_p=0.95, greedy=False)}\n")
    print(f"【設定: greedy】\n{engine.generate(q3, max_len=350, greedy=True)}\n")

    print("="*60)
    print("4. Penalty 推論測試 | 題目: 150字的 rap 歌詞")
    print("-" * 60)
    q4 = "請寫一段大約 150 字的 rap 歌詞，主題是關於寫程式。"
    print(f"【設定: 無】\n{engine.generate(q4, max_len=250, temp=0.8, top_p=0.9)}\n")
    # 將極端的懲罰數值調回合理區間
    print(f"【設定: repetition (rep=1.15)】\n{engine.generate(q4, max_len=250, temp=0.8, top_p=0.9, rep_p=1.15)}\n")
    print(f"【設定: frequency (freq=0.15)】\n{engine.generate(q4, max_len=250, temp=0.8, top_p=0.9, freq_p=0.15)}\n")
    print(f"【設定: presence (pres=0.15)】\n{engine.generate(q4, max_len=250, temp=0.8, top_p=0.9, pres_p=0.15)}\n")

    print("="*60)
    print("5. 進階策略 | 題目: 如果貓會說話，他會怎麼用 rap 的方式跟你介紹他的世界？(100字)")
    print("-" * 60)
    q5 = "如果貓會說話，他會怎麼用 rap 的方式跟你介紹他的世界？請寫大約 100 字的故事。"
    print(f"【策略 1: 創意意識流】\n{engine.generate(q5, max_len=250, temp=1.1, top_p=0.9, pres_p=0.2)}\n")
    print(f"【策略 2: 嚴謹敘事】\n{engine.generate(q5, max_len=250, temp=0.4, top_k=20, rep_p=1.1)}\n")
    print(f"【策略 3: 動態平衡】\n{engine.generate(q5, max_len=250, temp=0.75, min_p=0.05, freq_p=0.1)}\n")
    print("="*60)

if __name__ == "__main__":
    run_all_tasks()
