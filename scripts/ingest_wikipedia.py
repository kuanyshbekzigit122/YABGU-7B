"""
Kazakh Wikipedia (kk.wikipedia.org) Ingestion Script
Бұл скрипт қазақша Уикипедияның ашық лицензиялы (CC BY-SA 4.0) мақалаларын
жүктеп, бірыңғай JSONL форматына сақтайды.
"""
import sys
import os
import json
import re

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def ingest_wiki(max_articles=50000, output_file="data/raw/wiki_kk.jsonl"):
    print("=" * 80)
    print(" ҚАЗАҚША УИКИПЕДИЯ (WIKIPEDIA-KK) ДЕРЕКТЕРІН ЖҮКТЕУ ЖӘНЕ ӨҢДЕУ")
    print(" Лицензия: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ҚАТЕ] 'datasets' кітапханасы табылмады. 'pip install datasets' орындаңыз.")
        return

    print("[1/3] Hugging Face арқылы 'wikimedia/wikipedia' (kk) деректерін ағындық (streaming) режимде оқу...")
    try:
        # 20231101.kk немесе соңғы нұсқасы
        dataset = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
    except Exception as e:
        print(f"[ЕСКЕРТУ] 20231101.kk жүктелмеді: {e}. 'wikimedia/wikipedia' 2024 жылғы нұсқасын тексерудеміз...")
        dataset = load_dataset("wikimedia/wikipedia", "latest.kk", split="train", streaming=True)

    print(f"[2/3] Мақалаларды сүзу және тазалау (Максималды мақсат: {max_articles:,} мақала)...")
    
    count = 0
    total_tokens_approx = 0
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(dataset):
            title = item.get("title", "")
            raw_text = item.get("text", "")
            url = item.get("url", f"https://kk.wikipedia.org/wiki/{title}")
            
            # Өте қысқа мақалаларды немесе бағыттауыштарды (redirects) өткізіп жіберу
            if len(raw_text.strip()) < 150:
                continue
            if "айрық" in title.lower() or "мағына" in title.lower():
                continue
                
            # Артық тақырыпшаларды және сілтемелерді реттеу
            cleaned_text = re.sub(r"\n\s*\n+", "\n\n", raw_text).strip()
            tokens_est = len(cleaned_text) // 5
            total_tokens_approx += tokens_est
            
            doc = {
                "id": f"wiki_kk_{count+1:06d}",
                "title": title,
                "text": cleaned_text,
                "source": "kk.wikipedia.org",
                "url": url,
                "license": "CC BY-SA 4.0",
                "language": "kk",
                "category": "encyclopedia",
                "chars_count": len(cleaned_text),
                "tokens_estimate": tokens_est
            }
            
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 2000 == 0:
                print(f"  -> {count:,} мақала өңделді | Сақталған токен көлемі: ~{total_tokens_approx:,}")
                
            if count >= max_articles:
                break

    print("[3/3] Жүктеу және бастапқы өңдеу аяқталды!")
    print("=" * 80)
    print(f"Жалпы сақталған мақала саны: {count:,}")
    print(f"Жалпы сақталған токен саны: ~{total_tokens_approx:,} токен")
    print(f"Файл: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    ingest_wiki(max_articles=max_count)
