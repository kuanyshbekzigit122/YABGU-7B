"""
YABGU Inference & Text Generation Script
Үйретілген YABGU моделін іске қосып, берілген қазақша промптты жалғастыру скрипті.
"""

import sys
import os
import torch
from tokenizers import Tokenizer

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def generate_text(
    prompt: str,
    checkpoint_path: str = "checkpoints/checkpoint_last.pt",
    tokenizer_path: str = "tokenizer/vocab/tokenizer.json",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Құрылғы (Device): {device.upper()}")
    
    if checkpoint_path == "checkpoints/checkpoint_last.pt" and os.path.exists("checkpoints/checkpoint_colab_1500.pt"):
        checkpoint_path = "checkpoints/checkpoint_colab_1500.pt"

    if not os.path.exists(checkpoint_path):
        print(f"[ҚАТЕ] Checkpoint файлы табылмады: {checkpoint_path}")
        print("Алдымен модельді оқытыңыз ('python training/trainer.py')")
        return

    # 1. Токенизаторды жүктеу
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 2. Модельді жүктеу
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", YabguConfig())
    model = YabguTransformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Модель сәтті жүктелді (Step: {ckpt.get('step', 'unknown')})")

    # 3. Промптты кодтау
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    print("\n" + "=" * 80)
    print(f"ПРОМПТ: {prompt}")
    print("=" * 80)

    # 4. Генерация
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # 5. Декодтау
    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    print(f"YABGU ЖАУАБЫ:\n{generated_text}")
    print("=" * 80)

if __name__ == "__main__":
    prompt_input = "Қазақстанның елордасы — "
    if len(sys.argv) > 1:
        prompt_input = " ".join(sys.argv[1:])
    generate_text(prompt=prompt_input)
