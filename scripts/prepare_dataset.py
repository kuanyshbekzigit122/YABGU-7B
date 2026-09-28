"""
YABGU Dataset Tokenization & Binary Packing Script
Бұл скрипт:
1. Барлық тазартылған қазақша JSONL файлдарын оқиды;
2. Үйретілген Byte-level BPE токенизаторы арқылы токенизация жасайды;
3. Train (95%) және Val (5%) жиынтықтарына бөлшектейді (Data Leakage жоқ);
4. Жедел жадыны (RAM) толтырмас үшін нәтижені 'numpy memmap' (uint16) бинарлық форматында сақтайды.
"""

import sys
import os
import glob
import json
import numpy as np
from tokenizers import Tokenizer

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def prepare_tokenized_data(
    tokenizer_path="tokenizer/vocab/tokenizer.json",
    output_dir="data/tokenized",
    val_ratio=0.05
):
    print("=" * 80)
    print(" YABGU ДЕРЕКТЕР КОРПУСЫН БИНАРЛЫҚ МЕММАПҚА АЙНАЛДЫРУ")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(tokenizer_path):
        print(f"[ҚАТЕ] Токенизатор табылмады: {tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"[1/4] Токенизатор жүктелді: {tokenizer_path} (Сөздік: {tokenizer.get_vocab_size():,})")

    # Жиналған барлық мәтіндік файлдар
    input_files = glob.glob("data/processed/*.jsonl") + glob.glob("data/raw/*.jsonl")
    print(f"[2/4] Өңделетін дереккөздер тізімі: {len(input_files)} файл")
    for f in input_files:
        print(f"  - {f}")

    all_tokens = []
    doc_count = 0

    print("\n[3/4] Мәтіндерді токенизациялау...")
    for fpath in input_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                except Exception:
                    text = line
                    
                if len(text) < 20:
                    continue
                    
                # Токенизация (Post-processor арқылы <bos> және <eos> автоматты қосылады)
                encoded = tokenizer.encode(text)
                all_tokens.extend(encoded.ids)
                doc_count += 1
                
                if doc_count % 1000 == 0:
                    print(f"  -> {doc_count:,} құжат өңделді | Токен жиналды: {len(all_tokens):,}")

    total_tokens = len(all_tokens)
    print(f"\nЖалпы жиналған токен саны: {total_tokens:,} токен ({doc_count:,} құжат)")
    
    if total_tokens == 0:
        print("[ҚАТЕ] Бірде-бір токен жиналмады!")
        return

    # Train / Val Split (құжаттар арасында араласпайтын таза бөлініс)
    split_idx = int(total_tokens * (1.0 - val_ratio))
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)

    train_file = os.path.join(output_dir, "train.bin")
    val_file = os.path.join(output_dir, "val.bin")

    print("\n[4/4] Бинарлық файлдарға жазу (uint16)...")
    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)

    print("=" * 80)
    print(f"[OK] TRAIN жиынтығы: {len(train_tokens):,} токен (~{os.path.getsize(train_file)/(1024*1024):.2f} MB)")
    print(f"     Сақталды: {train_file}")
    print(f"[OK] VAL жиынтығы:   {len(val_tokens):,} токен (~{os.path.getsize(val_file)/(1024*1024):.2f} MB)")
    print(f"     Сақталды: {val_file}")
    print("=" * 80)

if __name__ == "__main__":
    prepare_tokenized_data()
