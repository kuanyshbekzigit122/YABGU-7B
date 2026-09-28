import sys
import subprocess

print("Installing datasets...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)

from datasets import load_dataset
print("Testing streaming wikipedia kk...")
try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
    count = 0
    for item in ds:
        print(f"Sample {count+1}: Title='{item.get('title')}', Length={len(item.get('text'))}")
        count += 1
        if count >= 3:
            break
    print("[SUCCESS] Wikipedia streaming works perfectly!")
except Exception as e:
    print("[ERROR] Streaming failed:", e)
