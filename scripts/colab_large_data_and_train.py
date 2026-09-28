"""
YABGU Large-Scale Pretraining Pipeline (Google Colab T4)
Кезеңдері:
1. /yabgu_pkg.zip архивін ашып, барлық модульдер мен кітаптарды /content ішіне дайындау;
2. Google Cloud желісінің жоғары жылдамдығын (1 Gbps+) пайдаланып, 15,000 Уикипедия мақаласы
   мен ALASH кітаптарын біріктіріп, үлкен тазартылған корпус (/content/data/processed/corpus_large.jsonl) жасау (~12M-15M токен);
3. 16,384 Byte-level BPE токенизаторын осы үлкен корпуспен үйрету;
4. Барлық токендерді бинарлық uint16 memmap файлына көшіру;
5. NVIDIA Tesla T4 GPU-да 1,500 қадамдық ірі pretraining өткізу;
6. Тілдік сапалық бағалауды (Inference) жүргізу.
"""

import sys
import os
import time
import zipfile
import re
import json
import unicodedata
import subprocess

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# 1. Жобаны дайындау
for p in ["/content", "."]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    if os.path.exists("/content"):
        os.chdir("/content")
except Exception:
    pass

print("=" * 80)
print(" YABGU ІРІ АУҚЫМДЫ ДЕРЕКТЕР ЖИНАУ ЖӘНЕ ОҚЫТУ ПАЙПЛАЙНЫ")
print("=" * 80)

# 1.1 yabgu_pkg.zip архивін ашу
zip_candidates = ["/yabgu_pkg.zip", "/content/yabgu_pkg.zip", "yabgu_pkg.zip"]
for zc in zip_candidates:
    if os.path.exists(zc):
        print(f"[SETUP] '{zc}' табылды, /content ішіне шығарылуда...")
        with zipfile.ZipFile(zc, "r") as z:
            z.extractall("/content")
        print("[SETUP] Барлық модульдер мен кітаптар сәтті шығарылды!")
        break

# 1.2 Қажетті кітапханаларды тексеру және орнату
print("\n[1/5] Қажетті кітапханаларды Colab жүйесінде тексеру/орнату...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers", "pyarrow", "fasttext-wheel"], check=False)

import torch
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Деректерді жинау (Ingestion)
print("\n[2/5] ҚАЗАҚША УИКИПЕДИЯ ЖӘНЕ ALASH КІТАПТАРЫНАН ҮЛКЕН КОРПУС ЖАСАУ...")
from datasets import load_dataset

os.makedirs("/content/data/raw", exist_ok=True)
os.makedirs("/content/data/processed", exist_ok=True)
output_corpus = "/content/data/processed/corpus_large.jsonl"

def clean_kz_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[←\d+\][^\n]*", "", text)
    text = re.sub(r"\b\d+\s+Әдеби\s+KZ\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # І/і гомоглифтерін түзету (латын i/I -> қазақша і/І)
    text = re.sub(r"(?<=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])[iI](?=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])", "і", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

doc_count = 0
total_chars = 0

with open(output_corpus, "w", encoding="utf-8") as f_out:
    # A) ALASH кітаптары
    alash_clean_path = "/content/data/processed/alash_clean.jsonl"
    if os.path.exists(alash_clean_path):
        with open(alash_clean_path, "r", encoding="utf-8") as f_alash:
            for line in f_alash:
                if line.strip():
                    f_out.write(line.strip() + "\n")
                    doc_count += 1
                    data = json.loads(line)
                    total_chars += len(data.get("text", ""))
        print(f"  -> ALASH кітаптары корпуске қосылды (Құжат саны: {doc_count})")

    # B) Уикипедия (15,000 сапалы мақала ағындық режимде)
    target_articles = 15000
    print(f"  -> Қазақша Уикипедия мақалалары жүктелуде (Мақсат: {target_articles:,} мақала)...")
    try:
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
        wiki_count = 0
        for item in wiki_ds:
            raw_text = item.get("text", "")
            title = item.get("title", "")
            if len(raw_text) < 150 or "айрық" in title.lower():
                continue
            cleaned = clean_kz_text(raw_text)
            tokens_est = len(cleaned) // 5
            
            doc = {
                "id": f"wiki_large_{wiki_count+1:06d}",
                "title": title,
                "text": cleaned,
                "source": "kk.wikipedia.org",
                "license": "CC BY-SA 4.0",
                "language": "kk",
                "tokens_estimate": tokens_est
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            doc_count += 1
            wiki_count += 1
            total_chars += len(cleaned)
            
            if wiki_count % 2500 == 0:
                print(f"     Уикипедия: {wiki_count:,}/{target_articles:,} мақала өңделді | Шамамен токен: ~{total_chars//5:,}")
            if wiki_count >= target_articles:
                break
    except Exception as e:
        print(f"  [ЕСКЕРТУ] Уикипедия жүктеуде қате немесе шектеу: {e}")

print(f"[OK] Үлкен корпус дайын: {doc_count:,} құжат | Шамамен ~{total_chars//5:,} токен!")

# 3. Токенизаторды оқыту
print("\n[3/5] 16K BYTE-LEVEL BPE ТОКЕНИЗАТОРЫН ҮЛКЕН КОРПУСТА ҚАЙТА ҮЙРЕТУ...")
from tokenizer.train_tokenizer import train_kazakh_tokenizer

tok_dir = "/content/tokenizer/vocab"
os.makedirs(tok_dir, exist_ok=True)
train_kazakh_tokenizer([output_corpus], output_dir=tok_dir, vocab_size=16384)

# 4. Бинарлық Memmap жинақтау
print("\n[4/5] ДЕРЕКТЕРДІ БИНАРЛЫҚ МЕММАПҚА (train.bin, val.bin) КӨШІРУ...")
import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
all_tokens = []

with open(output_corpus, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get("text", "")
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
        if (idx + 1) % 5000 == 0:
            print(f"  -> Токенизацияланды: {idx+1:,} құжат | Токен: {len(all_tokens):,}")

total_tok = len(all_tokens)
print(f"[OK] Жалпы жиналған бинарлық токен: {total_tok:,} токен!")

val_len = int(total_tok * 0.05)
train_tokens = np.array(all_tokens[:-val_len], dtype=np.uint16)
val_tokens = np.array(all_tokens[-val_len:], dtype=np.uint16)

os.makedirs("/content/data/tokenized", exist_ok=True)
train_bin = "/content/data/tokenized/train.bin"
val_bin = "/content/data/tokenized/val.bin"

train_tokens.tofile(train_bin)
val_tokens.tofile(val_bin)
print(f"  Train: {len(train_tokens):,} токен ({os.path.getsize(train_bin)/(1024*1024):.1f} MB)")
print(f"  Val:   {len(val_tokens):,} токен ({os.path.getsize(val_bin)/(1024*1024):.1f} MB)")

# 5. Colab T4 GPU-да 1,500 қадамдық Pretraining
print("\n[5/5] PRETRAINING: 1,500 ҚАДАМДЫҚ НЕГІЗГІ ОҚЫТУ БАСТАЛДЫ...")
from configs.yabgu_50m import YabguConfig
from training.trainer import train

train_config = YabguConfig(
    max_seq_len=256,
    micro_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=6e-4,
    min_lr=6e-5,
    mixed_precision="fp16"
)

train(
    max_steps=1500,
    eval_interval=100,
    eval_iters=25,
    save_interval=500,
    checkpoint_dir="/content/checkpoints",
    resume=False,
    config=train_config
)

# 6. Финалдық Тексеру (Inference)
print("\n" + "=" * 80)
print(" ЖАҢА ОҚЫТЫЛҒАН YABGU-50M МОДЕЛІН ТЕКСЕРУ (INFERENCE):")
print("=" * 80)
from scripts.generate import generate_text

eval_prompts = [
    "Қазақстанның елордасы — ",
    "Қазақ тілі — ",
    "Абай Құнанбайұлы — қазақтың ұлы ",
    "Ғылым мен білімнің маңызы ",
    "Алматы қаласының табиғаты "
]

for p in eval_prompts:
    print(f"\n--- Промпт: '{p}' ---")
    generate_text(
        prompt=p,
        checkpoint_path="/content/checkpoints/checkpoint_last.pt",
        tokenizer_path=os.path.join(tok_dir, "tokenizer.json"),
        max_new_tokens=45,
        temperature=0.7,
        top_k=40
    )

print("\n" + "=" * 80)
print(" ІРІ АУҚЫМДЫ PRETRAINING ЖӘНЕ БАҒАЛАУ ТОЛЫҚ АЯҚТАЛДЫ! ")
print("=" * 80)
