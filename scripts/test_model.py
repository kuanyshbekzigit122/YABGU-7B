"""
YABGU-50M Architecture Verification Script
Бұл скрипт құрастырылған PyTorch Transformer моделін тестілейді:
1. Нақты параметрлер санын есептейді;
2. Forward pass және Cross-Entropy Loss-ты тексереді;
3. Мәтін генерациясының (inference) логикасын тексереді.
"""

import sys
import os
import torch

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

def test_model():
    print("=" * 80)
    print(" YABGU-50M МОДЕЛЬ АРХИТЕКТУРАСЫН ТЕКСЕРУ")
    print("=" * 80)

    config = YabguConfig()
    print(f"Конфигурация:")
    print(f"  - Hidden dimension (d_model): {config.dim}")
    print(f"  - Layers: {config.n_layers}")
    print(f"  - Attention heads: {config.n_heads} (head_dim = {config.dim // config.n_heads})")
    print(f"  - Context length: {config.max_seq_len} tokens")
    print(f"  - Intermediate FFN (SwiGLU): {config.intermediate_size}")
    print(f"  - Vocabulary size: {config.vocab_size:,}")
    print(f"  - Weight Tying: {config.tie_word_embeddings}")

    print("\n[1/3] Модельді инициализациялау...")
    model = YabguTransformer(config)
    
    param_count = model.count_parameters()
    print(f"[OK] Модель сәтті құрастырылды!")
    print(f"  >>> НАҚТЫ ПАРАМЕТРЛЕР САНЫ: {param_count:,} параметр (~{param_count / 1e6:.2f}M) <<<")
    
    # 2. Forward pass тесті
    print("\n[2/3] Forward Pass және Loss есептеуді тексеру...")
    bsz = 2
    seqlen = 64
    dummy_input = torch.randint(0, config.vocab_size, (bsz, seqlen))
    dummy_targets = torch.randint(0, config.vocab_size, (bsz, seqlen))

    logits, loss = model(dummy_input, targets=dummy_targets)
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Logits shape: {logits.shape} (Күтілген: [{bsz}, {seqlen}, {config.vocab_size}])")
    print(f"  - Causal Loss мәні: {loss.item():.4f}")
    assert logits.shape == (bsz, seqlen, config.vocab_size), "Logits өлшемі сәйкес емес!"
    assert loss is not None and not torch.isnan(loss), "Loss есептелмеді немесе NaN!"
    print("[OK] Forward pass және Loss есептеу мінсіз орындалды!")

    # 3. Генерация тесті
    print("\n[3/3] Авторегрессивті генерация (Inference) тексеру...")
    prompt_tokens = torch.tensor([[1, 4011, 15999]], dtype=torch.long)  # <bos> + 2 tokens
    generated = model.generate(prompt_tokens, max_new_tokens=10, temperature=0.8)
    print(f"  - Бастапқы prompt: {prompt_tokens.shape}")
    print(f"  - Генерацияланған sequence: {generated.shape}")
    print(f"  - Шыққан token ID-лер: {generated.tolist()[0]}")
    print("[OK] Генерация тетігі сәтті жұмыс істеп тұр!")

    print("\n" + "=" * 80)
    print(" ТЕСТ СӘТТІ АЯҚТАЛДЫ! YABGU-50M PRETRAINING-КЕ ДАЙЫН.")
    print("=" * 80)

if __name__ == "__main__":
    test_model()
