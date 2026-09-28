"""
Step 4 & 5: Pretraining YABGU-50M and Inference on NVIDIA Tesla T4 GPU
"""
import sys
import os
import time
import torch

sys.path.insert(0, "/content")
from configs.yabgu_50m import YabguConfig
from training.trainer import train
from scripts.generate import generate_text

print("=" * 80)
print(" YABGU-50M НЕГІЗГІ PRETRAINING БАСТАЛДЫ (TESLA T4 16GB GPU)")
print("=" * 80)

tok_dir = "/content/tokenizer/vocab"
tok_json = os.path.join(tok_dir, "tokenizer.json")

train_config = YabguConfig(
    vocab_size=16384,
    max_seq_len=256,
    dim=512,
    n_layers=12,
    n_heads=8,
    intermediate_size=1408,
    micro_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_steps=100,
    weight_decay=0.1,
    mixed_precision="fp16"
)

# 1. 1,500 қадамдық жаттығу циклі
train(
    max_steps=1500,
    eval_interval=100,
    eval_iters=25,
    save_interval=500,
    checkpoint_dir="/content/checkpoints",
    resume=False,
    config=train_config
)

# 2. Модельді тексеру және сапалық бағалау (Inference)
print("\n" + "=" * 80)
print(" PRETRAINED YABGU-50M МОДЕЛІН БАҒАЛАУ (INFERENCE МӘТІН ҚҰРАУ)")
print("=" * 80)

eval_prompts = [
    "Қазақстанның елордасы — ",
    "Қазақ тілі — ",
    "Абай Құнанбайұлы — қазақтың ұлы ",
    "Ғылым мен білімнің маңызы ",
    "Алматы қаласының табиғаты ",
]

last_ckpt = "/content/checkpoints/checkpoint_last.pt"
if not os.path.exists(last_ckpt):
    last_ckpt = "/content/checkpoints/checkpoint_step_01500.pt"

for p in eval_prompts:
    print(f"\n>>> СЫНАҚ ПРОМПТЫ: '{p}'")
    generate_text(
        prompt=p,
        checkpoint_path=last_ckpt,
        tokenizer_path=tok_json,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40
    )

print("\n" + "=" * 80)
print(" YABGU-50M PRETRAINING ЖӘНЕ ЭВАЛЮАЦИЯСЫ СӘТТІ АЯҚТАЛДЫ! ")
print("=" * 80)
