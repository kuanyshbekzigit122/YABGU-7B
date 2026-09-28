"""
YABGU Kazakh Linguistic & Knowledge Benchmark
Модельдің қазақ тілінің грамматикасын, сингармонизмін және сөздік қорын қаншалықты
меңгергенін бағалайтын арнайы тестілеу жүйесі.
"""

import sys
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

# Сингармонизм және морфологиялық тест жұптары
# [Контекст, Дұрыс жалғау/сөз, Қате жалғау/сөз]
MORPHOLOGY_PAIRS = [
    ("Мектеп", "ке", "ға"),      # Барыс септігі (жіңішке)
    ("Қала", "ға", "ге"),        # Барыс септігі (жуан)
    ("Бала", "лар", "лер"),      # Көптік жалғау (жуан)
    ("Үй", "лер", "лар"),        # Көптік жалғау (жіңішке)
    ("Дос", "тың", "тің"),       # Ілік септігі (жуан, қатаң)
    ("Кітап", "тар", "тер"),     # Көптік (жуан)
]

KNOWLEDGE_PROMPTS = [
    ("Қазақстанның елордасы — ", ["Астана", "қаласы"]),
    ("Абай Құнанбайұлы — қазақтың ұлы ", ["ақыны", "ақын"]),
    ("Жетісу жерінде орналасқан үлкен көл — ", ["Балқаш", "Алакөл"]),
]

def evaluate_linguistics(
    checkpoint_path="checkpoints/checkpoint_last.pt",
    tokenizer_path="tokenizer/vocab/tokenizer.json"
):
    print("=" * 80)
    print(" YABGU ҚАЗАҚ ТІЛДІК ГРАММАТИКА МЕН СИНТАКСИС БЕНЧМАРКІ")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(checkpoint_path):
        print(f"[ҚАТЕ] Checkpoint файлы табылмады: {checkpoint_path}")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", YabguConfig())
    model = YabguTransformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[1/2] Морфологиялық сингармонизм тесті...")
    correct_count = 0
    
    for prefix, correct_suffix, wrong_suffix in MORPHOLOGY_PAIRS:
        encoded_prompt = tokenizer.encode(prefix).ids
        input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :]  # Соңғы токеннің ықтималдығы
            
        corr_id = tokenizer.token_to_id(correct_suffix) or tokenizer.token_to_id(" " + correct_suffix)
        wrong_id = tokenizer.token_to_id(wrong_suffix) or tokenizer.token_to_id(" " + wrong_suffix)
        
        if corr_id is not None and wrong_id is not None:
            prob_corr = next_logits[corr_id].item()
            prob_wrong = next_logits[wrong_id].item()
            is_correct = prob_corr > prob_wrong
            if is_correct:
                correct_count += 1
            status = "ДҰРЫС (+)" if is_correct else "ҚАТЕ (-)"
            print(f"  {prefix:<10} -> '{correct_suffix}' vs '{wrong_suffix}' | {status}")
        else:
            print(f"  {prefix:<10} -> Токендер сөздіктен табылмады.")

    print(f"\nСингармонизм дәлдігі: {correct_count}/{len(MORPHOLOGY_PAIRS)} ({correct_count/len(MORPHOLOGY_PAIRS)*100:.1f}%)")

    print("\n[2/2] Білім және логикалық жалғастыру үлгілері...")
    for prompt, expected in KNOWLEDGE_PROMPTS:
        encoded = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=15, temperature=0.7)
        res = tokenizer.decode(out[0].tolist())
        print(f"  Промпт: '{prompt}'")
        print(f"  Нәтиже: {res.strip()}\n")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_linguistics()
