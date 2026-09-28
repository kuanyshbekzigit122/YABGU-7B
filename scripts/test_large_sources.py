import os
import sys

print("Checking available Kazakh dataset sources on HuggingFace...")
from datasets import load_dataset

# 1. Wikipedia KK
try:
    print("[1] Testing Wikipedia kk...")
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
    for idx, item in enumerate(ds_wiki):
        if idx >= 2:
            break
        print(f"  Wiki #{idx+1}: {item.get('title')[:30]} ({len(item.get('text'))} chars)")
    print("  -> Wikipedia KK: OK")
except Exception as e:
    print("  -> Wikipedia KK Error:", e)

# 2. CulturaX or mC4 or Oscar
try:
    print("[2] Testing mc4 kk...")
    ds_c4 = load_dataset("allenai/c4", "kk", split="train", streaming=True)
    for idx, item in enumerate(ds_c4):
        if idx >= 2:
            break
        print(f"  mC4 #{idx+1}: {item.get('text')[:60]}... ({len(item.get('text'))} chars)")
    print("  -> mC4 KK: OK")
except Exception as e:
    print("  -> mC4 KK Error:", e)
