"""
Evaluate Trained YABGU-50M Model on Colab T4
Бұл скрипт Colab-тағы ең соңғы жаттыққан 'checkpoint_last.pt' салмақтарын жүктеп,
қазақша грамматика, сингармонизм және сөйлем құрау мүмкіндігін бағалайды.
"""

import os
import sys
import torch

for p in ["/content", "."]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    if os.path.exists("/content"):
        os.chdir("/content")
except Exception:
    pass

from scripts.generate import generate_text
from evaluation.kazakh_benchmarks import evaluate_linguistics

ckpt_path = "/content/checkpoints/checkpoint_last.pt" if os.path.exists("/content/checkpoints/checkpoint_last.pt") else "checkpoints/checkpoint_last.pt"
tok_path = "/content/tokenizer/vocab/tokenizer.json" if os.path.exists("/content/tokenizer/vocab/tokenizer.json") else "tokenizer/vocab/tokenizer.json"

print("=" * 80)
print(f" YABGU-50M СОҢҒЫ CHECKPOINT БОЙЫНША ҚАЗАҚША БАҒАЛАУ: {ckpt_path}")
print("=" * 80)

# 1. Грамматикалық тест
evaluate_linguistics(checkpoint_path=ckpt_path, tokenizer_path=tok_path)

# 2. Мәтін генерациясы
test_prompts = [
    "Қазақстанның астанасы — ",
    "Абай Құнанбайұлының қара сөздері ",
    "Мектепте балалар ",
    "Ғылым мен білім ",
]

print("\n" + "=" * 80)
print(" МӘТІН ТУДЫРУ ТЕСТІ (TEXT GENERATION SAMPLES):")
print("=" * 80)

for prompt in test_prompts:
    print(f"\n--- Промпт: {prompt} ---")
    generate_text(
        prompt=prompt,
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        max_new_tokens=40,
        temperature=0.7,
        top_k=40
    )
