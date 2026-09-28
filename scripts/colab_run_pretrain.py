"""
YABGU Colab Pretraining Runner
Бұл скрипт Google Colab ортасында NVIDIA Tesla T4 GPU арқылы орындалады:
1. 'yabgu_pkg.zip' архивін ашады;
2. GPU-ны тексереді;
3. YABGU-50M моделін 1000 қадам бойы pretraining жасайды (Mixed Precision fp16);
4. Жаттығу аяқталған соң сақталған checkpoint арқылы қазақша мәтін генерациясын тексереді.
"""

import os
import sys
import zipfile
import time

# 1. Архивтi ашу
zip_path = "/content/yabgu_pkg.zip" if os.path.exists("/content/yabgu_pkg.zip") else "yabgu_pkg.zip"
if os.path.exists(zip_path):
    print(f"[1/4] Жоба файлдары архивінен ашылуда: {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")
    print("[OK] Барлық файлдар /content ішіне сәтті орнатылды.")

sys.path.insert(0, "/content")
try:
    os.chdir("/content")
except Exception:
    pass

# 2. Жүйені тексеру
import torch
print("\n[2/4] GPU ТЕКСЕРІСІ:")
print(f"PyTorch: {torch.__version__} | CUDA қолжетімді ме: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU құрылғысы: {torch.cuda.get_device_name(0)}")
    print(f"VRAM көлемі: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[ЕСКЕРТУ] GPU анықталмады, CPU қолданылуда!")

# 3. Модель конфигурациясы мен жаттығу
print("\n[3/4] PRETRAINING БАСТАЛДЫ...")
sys.path.insert(0, ".")

from training.trainer import train

# 500 қадамдық pretraining (Colab T4-те шамамен 2 минут алады)
train(
    max_steps=500,
    eval_interval=50,
    eval_iters=10,
    save_interval=100,
    checkpoint_dir="checkpoints",
    resume=True
)

# 4. Тексеру және Қазақша мәтін генерациясы
print("\n[4/4] ЖАТТЫҚҚАН МОДЕЛЬДІ ТЕКСЕРУ (INFERENCE):")
from scripts.generate import generate_text

test_prompts = [
    "Қазақстанның елордасы — ",
    "Абай Құнанбайұлының «Қара сөздері» ",
    "Балалар мектепке ",
]

for p in test_prompts:
    print("-" * 60)
    generate_text(prompt=p, checkpoint_path="checkpoints/checkpoint_last.pt", max_new_tokens=40, temperature=0.7)

print("=" * 80)
print(" YABGU-50M PRETRAINING & EVALUATION СӘТТІ АЯҚТАЛДЫ! ")
print("=" * 80)
