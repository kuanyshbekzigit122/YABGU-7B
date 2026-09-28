"""
Step 1: Large-Scale Ingestion for YABGU (Kazakh Wikipedia + ALASH Books)
"""
import sys
import os
import re
import json
import unicodedata
import subprocess

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

print("=" * 80)
print(" YABGU ДЕРЕКТЕР КОРПУСЫН ДАЙЫНДАУ (INGESTION)")
print("=" * 80)

# Check datasets
try:
    import datasets
except ImportError:
    print("[1/2] 'datasets' орнатылуда...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
    import datasets

from datasets import load_dataset

os.makedirs("/content/data/processed", exist_ok=True)
output_corpus = "/content/data/processed/corpus_large.jsonl"

def clean_kz_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[←\d+\][^\n]*", "", text)
    text = re.sub(r"\b\d+\s+Әдеби\s+KZ\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?<=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])[iI](?=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])", "і", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

doc_count = 0
total_chars = 0

with open(output_corpus, "w", encoding="utf-8") as f_out:
    # 1. ALASH Books
    alash_path = "/content/data/processed/alash_clean.jsonl"
    if os.path.exists(alash_path):
        with open(alash_path, "r", encoding="utf-8") as f_alash:
            for line in f_alash:
                if line.strip():
                    f_out.write(line.strip() + "\n")
                    doc_count += 1
                    data = json.loads(line)
                    total_chars += len(data.get("text", ""))
        print(f"[ALASH] {doc_count} кітап толық қосылды ({total_chars:,} таңба).")
    else:
        print("[ALASH] Ескерту: alash_clean.jsonl табылмады.")

    # 2. Wikipedia kk streaming (12,000 articles)
    target_articles = 12000
    print(f"[WIKI] Қазақша Уикипедиядан {target_articles:,} сапалы мақала жүктелуде...")
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
        
        if wiki_count % 3000 == 0:
            print(f"  -> Уикипедия: {wiki_count:,}/{target_articles:,} мақала | Барлық таңба: ~{total_chars:,} | Токен: ~{total_chars//5:,}")
        if wiki_count >= target_articles:
            break

file_size_mb = os.path.getsize(output_corpus) / (1024 * 1024)
print("=" * 80)
print(f"[НӘТИЖЕ] Корпус дайын: {output_corpus}")
print(f"Жалпы құжаттар: {doc_count:,} (Уикипедия: {wiki_count:,} + ALASH: {doc_count - wiki_count})")
print(f"Жалпы көлемі: {file_size_mb:.2f} MB | Шамамен токен: ~{total_chars//5:,}")
print("=" * 80)
