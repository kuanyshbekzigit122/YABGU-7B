"""
ALASH Books Deep Legal & Quality Audit Script
Әр кітаптың авторын, жылын, авторлық құқық мәртебесін және сапасын нақты тексереді.
"""
import sys
import os
import glob
import re
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Қазақ төл әріптері
KZ_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# Белгілі авторлар базасы (қайтыс болған жылы)
# ҚР Заңы бойынша 70 жыл ережесі (2026 жылы: 2026 - 70 = 1956 жылға дейін қайтыс болғандар - Public Domain)
KNOWN_AUTHORS = {
    "міржақып": {"full_name": "Міржақып Дулатұлы", "death": 1935, "status": "PUBLIC_DOMAIN"},
    "дулатұлы": {"full_name": "Міржақып Дулатұлы", "death": 1935, "status": "PUBLIC_DOMAIN"},
    "абай": {"full_name": "Абай Құнанбайұлы", "death": 1904, "status": "PUBLIC_DOMAIN"},
    "шәкәрім": {"full_name": "Шәкәрім Құдайбердіұлы", "death": 1931, "status": "PUBLIC_DOMAIN"},
    "ахмет байтұрсын": {"full_name": "Ахмет Байтұрсынұлы", "death": 1937, "status": "PUBLIC_DOMAIN"},
    "мағжан": {"full_name": "Мағжан Жұмабаев", "death": 1938, "status": "PUBLIC_DOMAIN"},
    "жүсіпбек": {"full_name": "Жүсіпбек Аймауытов", "death": 1930, "status": "PUBLIC_DOMAIN"},
    "сәкен сейфуллин": {"full_name": "Сәкен Сейфуллин", "death": 1938, "status": "PUBLIC_DOMAIN"},
    "ыбырай алтынсарин": {"full_name": "Ыбырай Алтынсарин", "death": 1889, "status": "PUBLIC_DOMAIN"},
    "дулат исабеков": {"full_name": "Дулат Исабеков", "death": None, "status": "DO_NOT_TRAIN_COPYRIGHTED"}, # Көзі тірі (1942 ж.т.)
    "төлен әбдіков": {"full_name": "Төлен Әбдіков", "death": None, "status": "DO_NOT_TRAIN_COPYRIGHTED"}, # Көзі тірі (1942 ж.т.)
    "бодо шефер": {"full_name": "Bodo Schäfer", "death": None, "status": "DO_NOT_TRAIN_COPYRIGHTED"}, # Шетелдік аударма
    "bodo schäfer": {"full_name": "Bodo Schäfer", "death": None, "status": "DO_NOT_TRAIN_COPYRIGHTED"},
    "санжар": {"full_name": "Санжар (заманауи автор)", "death": None, "status": "UNKNOWN_DO_NOT_TRAIN"},
}

def analyze_book(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    total_len = len(content)
    kz_count = sum(1 for c in content if c in KZ_CHARS)
    kz_ratio = (kz_count / total_len * 100) if total_len > 0 else 0
    
    first_2000 = content[:2000].lower()
    
    # Авторды іздеу
    detected_author = None
    author_status = "UNKNOWN_DO_NOT_TRAIN"
    
    for key, info in KNOWN_AUTHORS.items():
        if key in first_2000:
            detected_author = info["full_name"]
            author_status = info["status"]
            break
            
    # Егер автор анықталмаса, тақырыптан қарау
    lines = [l.strip() for l in content.splitlines() if len(l.strip()) > 0]
    sample_title = lines[0] if lines else "Бос файл"
    
    # OCR қателерін тексеру
    ocr_issues = []
    if "нъ" in content.lower():
        ocr_issues.append("'ң' орнына 'нъ'")
    if "к," in content.lower() or "к`" in content.lower():
        ocr_issues.append("'қ' орнына артефакт")
    if re.search(r"[а-яА-ЯёЁ][a-zA-Z][а-яА-ЯёЁ]", content):
        ocr_issues.append("Кирилл/Латын араласуы")
        
    return {
        "file": os.path.basename(filepath),
        "size_kb": os.path.getsize(filepath) / 1024,
        "chars": total_len,
        "title": sample_title[:60],
        "author": detected_author or "Белгісіз",
        "legal_status": author_status,
        "kz_char_pct": round(kz_ratio, 2),
        "ocr_issues": ocr_issues
    }

def main():
    books = sorted(glob.glob("data/raw/alash-data/books/*.md"))
    results = [analyze_book(b) for b in books]
    
    print("=" * 100)
    print(" ALASH CDN КІТАПТАРЫНЫҢ ҚҰҚЫҚТЫҚ ЖӘНЕ САПАЛЫҚ АУДИТІ")
    print("=" * 100)
    
    safe_to_train = []
    do_not_train = []
    
    for r in results:
        status_tag = "[РҰҚСАТ: ОҚЫТУҒА БОЛАДЫ]" if r["legal_status"] == "PUBLIC_DOMAIN" else "[ТЫЙЫМ: DO NOT TRAIN]"
        if r["legal_status"] == "PUBLIC_DOMAIN":
            safe_to_train.append(r)
        else:
            do_not_train.append(r)
            
        print(f"\nФайл: {r['file']}")
        print(f"  Атауы: {r['title']}")
        print(f"  Автор: {r['author']}")
        print(f"  Құқықтық мәртебесі: {status_tag} ({r['legal_status']})")
        print(f"  Көлемі: {r['size_kb']:.1f} KB | Символ: {r['chars']:,} | Қазақ әріптерінің үлесі: {r['kz_char_pct']}%")
        if r["ocr_issues"]:
            print(f"  OCR мәселелері: {', '.join(r['ocr_issues'])}")
        else:
            print("  OCR мәселелері: Таза мәтін")

    print("\n" + "=" * 100)
    print(" АУДИТ ҚОРЫТЫНДЫСЫ:")
    print("=" * 100)
    print(f"Жалпы тексерілген кітап: {len(results)}")
    print(f"  - PUBLIC DOMAIN (Оқытуға заңды рұқсат): {len(safe_to_train)} кітап")
    print(f"  - DO NOT TRAIN / UNKNOWN (Авторлық құқықпен қорғалған немесе белгісіз): {len(do_not_train)} кітап")
    print("=" * 100)

if __name__ == "__main__":
    main()
