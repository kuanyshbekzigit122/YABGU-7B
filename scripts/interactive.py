"""
YABGU Interactive Console (Тікелей модельмен сөйлесу режимі)
Терминалда қазақша промпт енгізіп, YABGU-дың қалай жауап қайтаратынын интерактивті көруге арналған.
"""

import sys
import os
import torch
from tokenizers import Tokenizer

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

def start_interactive_session():
    # Ең соңғы checkpoint-ті таңдау
    ckpt_paths = [
        "checkpoints/checkpoint_colab_1500.pt",
        "checkpoints/checkpoint_colab_last.pt",
        "checkpoints/checkpoint_last.pt"
    ]
    ckpt_file = None
    for p in ckpt_paths:
        if os.path.exists(p):
            ckpt_file = p
            break
            
    if not ckpt_file:
        print("[ҚАТЕ] Checkpoint файлы табылмады!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" YABGU-50M ИНТЕРАКТИВТІ СӨЙЛЕСУ РЕЖИМІ")
    print(f" Құрылғы: {device.upper()} | Checkpoint: {ckpt_file}")
    print(" Шығу үшін 'exit' немесе 'шығу' деп жазыңыз.")
    print("=" * 70)

    # Токенизатор мен модельді жүктеу
    tokenizer = Tokenizer.from_file("tokenizer/vocab/tokenizer.json")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = ckpt.get("config", YabguConfig())
    
    model = YabguTransformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[OK] YABGU моделі дайын! (Оқытылған қадамы: {ckpt.get('step', 'unknown')})\n")

    while True:
        try:
            user_prompt = input("Сіз (промпт жазыңыз): ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ["exit", "quit", "шығу", "q"]:
                print("\nСессия аяқталды. Сау болыңыз!")
                break

            # Токенизация
            encoded = tokenizer.encode(user_prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

            # Генерация
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=45,
                    temperature=0.7,
                    top_k=40
                )

            res = tokenizer.decode(output_ids[0].tolist())
            print(f"\nYABGU: {res}\n")
            print("-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nСессия аяқталды.")
            break

if __name__ == "__main__":
    start_interactive_session()
