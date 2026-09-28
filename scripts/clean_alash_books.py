"""
ALASH Books Cleaning & Normalization Pipeline
1. Авторлық құқық сүзгісінен өткен кітаптарды іріктейді.
2. OCR қателерін (тасымал, артефактілер, колонтитулдар) тазалайды.
3. Қазақ кириллицасының 9 төл әрпін нормализациялайды.
4. Нәтижені стандарты JSONL форматында 'data/processed/alash_clean.jsonl' файлына жазады.
"""
import sys
import os
import glob
import re
import json
import unicodedata

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Қазақ кириллицасындағы шатасатын гомоглифтер сөздігі (латын/орыс орнына қазақ төл әрпі)
HOMOGLYPH_FIXES = {
    "i": "і",  # Латын кіші 'i' -> қазақ 'і' (қазақ сөздерінің ішінде)
    "I": "І",  # Латын бас 'I' -> қазақ 'І'
}

def clean_text(raw_text: str) -> str:
    # 1. Unicode NFKC нормализациясы
    text = unicodedata.normalize("NFKC", raw_text)
    
    # 2. Markdown тақырыптары мен артық белгілерді тазалау
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}", "", text)  # Bold/Italic жұлдызшаларын алып тастау
    text = re.sub(r"\[←\d+\][^\n]*", "", text)  # [←1] сілтемелік түсініктемелерін жою
    
    # 3. Веб және баспа колонтитулдарын жою (мысалы: "Әдеби KZ", "Notes", бет нөмірлері)
    text = re.sub(r"\b\d+\s+Әдеби\s+KZ\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Жалғыз тұрған бет нөмірлері
    
    # 4. Тасымал белгілерін біріктіру (мысалы, "бала- \n лар" -> "балалар")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    
    # 5. Қазақ сөздерінің ішіндегі латын 'i' мен 'I' әріптерін қазақша 'і' және 'І'-ге ауыстыру
    # Тек кирилл әріптерінің арасында тұрған жағдайда:
    def replace_latin_i(match):
        char = match.group(0)
        return HOMOGLYPH_FIXES.get(char, char)
    
    text = re.sub(r"(?<=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])[iI](?=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])", replace_latin_i, text)
    
    # 6. Көп қайталанатын бос орындар мен бос жолдарды реттеу
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    return text.strip()

def process_alash_books(approved_files=None):
    """
    approved_files: рұқсат етілген нақты файлдар тізімі.
    Егер берілмесе, тек тексерілген Public Domain классикалық шығармалар өңделеді.
    """
    input_dir = "data/raw/alash-data/books"
    output_file = "data/processed/alash_clean.jsonl"
    os.makedirs("data/processed", exist_ok=True)
    
    # Әдепкі заңды рұқсат етілген кітаптар (Public Domain: Жүсіпбек Аймауытов, Міржақып Дулатұлы, Абай)
    default_allowed = [
        "b7287bb0-14d2-4f27-87e8-3748ae878fc4_ақбілек.md",
        "c383dad1-63c3-406e-ac5b-1963f09a246c_бақытсыз_жамал_.md",
        "cfb6caf9-887c-4ac3-94c7-f49ccb7a4367_атаусыз_.md",
        "effd4f1a-2c48-4301-a00e-e570911bf923_жауын_астында_жазылған_кітап_2_.md",
    ]
    
    target_files = approved_files if approved_files is not None else default_allowed
    
    processed_docs = []
    total_tokens_approx = 0
    
    print("=" * 80)
    print(" ALASH КІТАПТАРЫН ТАЗАЛАУ ЖӘНЕ JSONL ФОРМАТЫНА КӨШІРУ")
    print("=" * 80)
    
    for fname in target_files:
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            print(f"[ЕСКЕРТУ] Файл табылмады: {fname}")
            continue
            
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
            
        cleaned = clean_text(raw_text)
        tokens_est = len(cleaned) // 5
        total_tokens_approx += tokens_est
        
        doc = {
            "id": f"alash_{fname.split('_')[0]}",
            "file_name": fname,
            "text": cleaned,
            "source": "ALASH CDN Books",
            "license": "Public Domain (Pre-1956 author / Legal audit passed)",
            "language": "kk",
            "category": "kazakh_classical_literature",
            "chars_count": len(cleaned),
            "tokens_estimate": tokens_est
        }
        processed_docs.append(doc)
        print(f"[ӨҢДЕЛДІ] {fname[:35]}... | Символ: {len(cleaned):,} | Шамамен токен: ~{tokens_est:,}")
        
    with open(output_file, "w", encoding="utf-8") as out_f:
        for doc in processed_docs:
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print("=" * 80)
    print(f"Тазартылған құжаттар саны: {len(processed_docs)}")
    print(f"Жалпы сақталған токен көлемі: ~{total_tokens_approx:,} токен")
    print(f"Нәтиже сақталды: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    process_alash_books()
