"""
Step 2 & 3: Tokenizer Training and Binary Memmap Packing on Colab
"""
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, "/content")
from tokenizer.train_tokenizer import train_kazakh_tokenizer
from tokenizers import Tokenizer

tok_dir = "/content/tokenizer/vocab"
corpus_path = "/content/data/processed/corpus_large.jsonl"

# 1. Train tokenizer
print("=" * 80)
print(" 16K BPE ТОКЕНИЗАТОРЫН ҮЛКЕН КОРПУСТА ОҚЫТУ")
print("=" * 80)
train_kazakh_tokenizer([corpus_path], output_dir=tok_dir, vocab_size=16384)

# 2. Tokenize and pack to memmap
print("\n" + "=" * 80)
print(" ДЕРЕКТЕРДІ БИНАРЛЫҚ МЕММАПҚА (train.bin, val.bin) КӨШІРУ")
print("=" * 80)

tokenizer = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
all_tokens = []
t0 = time.time()

with open(corpus_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get("text", "")
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
        if (idx + 1) % 3000 == 0:
            print(f"  -> Токенизацияланды: {idx+1:,} құжат | Токендер: {len(all_tokens):,}")

total_tok = len(all_tokens)
elapsed = time.time() - t0
print(f"[OK] Жалпы жиналған бинарлық токендер: {total_tok:,} ({elapsed:.1f} сек)")

val_len = int(total_tok * 0.05)
train_tokens = np.array(all_tokens[:-val_len], dtype=np.uint16)
val_tokens = np.array(all_tokens[-val_len:], dtype=np.uint16)

os.makedirs("/content/data/tokenized", exist_ok=True)
train_bin = "/content/data/tokenized/train.bin"
val_bin = "/content/data/tokenized/val.bin"

train_tokens.tofile(train_bin)
val_tokens.tofile(val_bin)

print(f"  Train: {len(train_tokens):,} токен ({os.path.getsize(train_bin)/(1024*1024):.2f} MB)")
print(f"  Val:   {len(val_tokens):,} токен ({os.path.getsize(val_bin)/(1024*1024):.2f} MB)")
print("=" * 80)
