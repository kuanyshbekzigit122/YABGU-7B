"""
YABGU-50M Instruct Model Verification Script
Қазақ тіліндегі нақты сұрақтар бойынша SFT моделін сынақтан өткізу.
"""

import sys
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

def load_instruct_model(ckpt_path="checkpoints/yabgu_50m_instruct_fp16.pt", device="cpu"):
    print(f"[*] Модель жүктелуде: {ckpt_path} ({device.upper()})...")
    config = YabguConfig()
    model = YabguTransformer(config)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    if device == "cpu":
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def ask_question(model, tokenizer, question, device="cpu", max_new_tokens=150, temperature=0.6, top_k=40, top_p=0.9):
    # Ensure post_processor doesn't append <|eos|> to the prompt!
    tokenizer.no_padding()
    tokenizer.post_processor = None
    
    bos_id = tokenizer.token_to_id("<|bos|>") or 1
    eos_id = tokenizer.token_to_id("<|eos|>") or 2
    
    prompt_str = f"### Сұрақ:\n{question.strip()}\n\n### Жауап:\n"
    encoded_ids = [bos_id] + tokenizer.encode(prompt_str).ids
    tokens = torch.tensor([encoded_ids], dtype=torch.long, device=device)
    
    stop_tokens = {eos_id}
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) >= model.config.max_seq_len:
                break
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            # Repetition penalty on recent tokens
            if len(generated) > 0:
                for prev_tok in set(generated[-20:]):
                    logits[0, prev_tok] /= 1.15
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = next_token.item()
            
            if next_id in stop_tokens:
                break
                
            generated.append(next_id)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            curr_text = tokenizer.decode(generated)
            if "###" in curr_text:
                curr_text = curr_text.split("###")[0]
                return curr_text.strip()
                
    return tokenizer.decode(generated).strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer/vocab/tokenizer.json")
    
    ckpt = "checkpoints/yabgu_50m_instruct_fp16.pt"
    if not os.path.exists(ckpt):
        print(f"Error: {ckpt} does not exist yet!")
        return
        
    model = load_instruct_model(ckpt, device=device)
    
    questions = [
        "Сен кімсің?",
        "Қазақстанның астанасы қандай қала?",
        "Абай Құнанбаев кім?",
        "Жасанды интеллект деген не?",
        "Дені сау болу үшін не істеу керек?"
    ]
    
    print("\n" + "=" * 70)
    print(" YABGU-50M INSTRUCT: СҰРАҚ-ЖАУАП ТЕСТІЛЕУ")
    print("=" * 70)
    
    for q in questions:
        print(f"\n[СҰРАҚ]: {q}")
        ans = ask_question(model, tokenizer, q, device=device)
        print(f"[ЖАУАП]:\n{ans}")
        print("-" * 50)

if __name__ == "__main__":
    main()
