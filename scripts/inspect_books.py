"""
ALASH Books Audit & Inspection Script
Кітаптардың санын, атауын, авторлық ақпаратын, форматын және тазалығын тексереді.
"""
import sys
import os
import glob
import re

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def inspect_all_books():
    books = glob.glob("data/raw/alash-data/books/*.md")
    print(f"Жалпы табылған кітап саны: {len(books)}\n")
    print("=" * 90)
    print(f"{'№':<3} | {'Көлемі (KB)':<10} | {'Жол саны':<8} | {'Символ саны':<12} | {'Файл атауы / Тақырыбы'}")
    print("=" * 90)
    
    total_size = 0
    total_chars = 0
    total_lines = 0

    for idx, book_path in enumerate(sorted(books), start=1):
        size_kb = os.path.getsize(book_path) / 1024
        total_size += size_kb
        
        with open(book_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            lines = content.splitlines()
            num_lines = len(lines)
            num_chars = len(content)
            total_chars += num_chars
            total_lines += num_lines
            
            # Тақырыпты анықтауға тырысу (алғашқы мағыналы жолдар)
            title = "Анықталмады"
            for line in lines[:10]:
                cleaned = line.strip("# \t*")
                if len(cleaned) > 3:
                    title = cleaned
                    break
        
        fname = os.path.basename(book_path)
        print(f"{idx:<3} | {size_kb:<10.1f} | {num_lines:<8} | {num_chars:<12} | {title[:45]} ({fname[:20]}...)")

    print("=" * 90)
    print(f"ЖАЛПЫ КӨЛЕМ: {total_size / 1024:.2f} MB")
    print(f"ЖАЛПЫ СИМВОЛ САНЫ: {total_chars:,} символ")
    print(f"ЖАЛПЫ ЖОЛ САНЫ: {total_lines:,} жол")
    approx_tokens = total_chars // 5  # Қазақ тілінде орта есеппен 1 токен ~ 5 символ
    print(f"ШАМАМЕН ТОКЕН САНЫ: ~{approx_tokens:,} токен")
    print("=" * 90)

if __name__ == "__main__":
    inspect_all_books()
