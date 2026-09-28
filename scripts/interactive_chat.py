"""
YABGU Interactive Chat REPL (Instruction / Assistant Mode)
Қазақ тіліндегі дербес YABGU моделімен тікелей диалог құру және сұрақ-жауап режимі.
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

def load_chat_model(checkpoint_path="checkpoints/yabgu_50m_instruct_fp16.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if not os.path.exists(checkpoint_path):
        # Fallback to scale3000 base checkpoint if instruct is not yet downloaded
        alt = "checkpoints/yabgu_50m_scale3000_fp16.pt"
        if os.path.exists(alt):
            print(f"[ЕСКЕРТУ] '{checkpoint_path}' әлі жоқ. Базалық модель қолданылады: {alt}")
            checkpoint_path = alt
        else:
            raise FileNotFoundError(f"Чекпоинт табылмады: {checkpoint_path}")
            
    print(f"[*] Модель жүктелуде: {checkpoint_path} ({device.upper()})...")
    config = YabguConfig()
    model = YabguTransformer(config)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Half to float if running on CPU
    if device == "cpu":
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config, device

def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.6,
    top_k: int = 35,
    top_p: float = 0.85
):
    tokenizer.no_padding()
    tokenizer.post_processor = None

    bos_id = tokenizer.token_to_id("<|bos|>") or 1
    eos_id = tokenizer.token_to_id("<|eos|>") or 2
    
    # Instruction template without trailing eos
    formatted_prompt = f"### Сұрақ:\n{prompt.strip()}\n\n### Жауап:\n"
    encoded = [bos_id] + tokenizer.encode(formatted_prompt).ids
    tokens = torch.tensor([encoded], dtype=torch.long, device=device)
    
    stop_tokens = {eos_id}
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) >= model.config.max_seq_len:
                break
                
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            # Repetition penalty
            if len(generated) > 0:
                for prev_tok in set(generated[-20:]):
                    logits[0, prev_tok] /= 1.15

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Top-P (Nucleus)
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
            
            # Check for stop sequence "###"
            current_text = tokenizer.decode(generated)
            if "###" in current_text:
                current_text = current_text.split("###")[0]
                return current_text.strip()
                
    response_text = tokenizer.decode(generated).strip()
    return response_text

def run_chat_repl():
    print("=" * 70)
    print(" YABGU-50M ҚАЗАҚША AI КӨМЕКШІСІ (INSTRUCT / CHAT РЕЖИМІ)")
    print(" Шығу үшін 'exit' немесе 'q' теріңіз.")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer/vocab/tokenizer.json")
    
    ckpt = "checkpoints/yabgu_50m_instruct_fp16.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/yabgu_50m_scale3000_fp16.pt"
        
    model, _, device = load_chat_model(ckpt, device=device)
    print("\n[Дайын!] Сұрағыңызды немесе тапсырмаңызды жазыңыз:\n")
    
    while True:
        try:
            user_input = input("Сіз: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Сау болыңыз!")
                break
                
            print("YABGU ойлануда...", end="\r", flush=True)
            answer = generate_answer(model, tokenizer, user_input, device)
            print(f"YABGU: {answer}\n")
        except KeyboardInterrupt:
            print("\nСессия аяқталды.")
            break
        except Exception as e:
            print(f"\n[Қате]: {e}")

if __name__ == "__main__":
    run_chat_repl()
