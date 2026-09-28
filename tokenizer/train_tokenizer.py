"""
YABGU Custom Kazakh Tokenizer Trainer
Архитектура: Byte-level BPE (Byte-Pair Encoding)
Сөздік көлемі: 16,384 tokens (YABGU-50M үшін оңтайландырылған)
Ерекшелігі: Қазақ тілінің агглютинативті құрылымын, түбір мен жалғауларын
және 9 төл әрпін (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І) 0% OOV (Out-of-Vocabulary) деңгейінде өңдейді.
"""

import sys
import os
import glob
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

# Арнайы токендер тізімі
SPECIAL_TOKENS = [
    "<|unk|>",   # 0: Белгісіз токен (Byte-level кезінде сирек қолданылады)
    "<|bos|>",   # 1: Мәтіннің басы (Beginning of Sequence)
    "<|eos|>",   # 2: Мәтіннің соңы (End of Sequence)
    "<|pad|>",   # 3: Padding токені
    "<|sep|>",   # 4: Аралық бөлгіш
]

def text_iterator(corpus_files, batch_size=2000):
    """Жадыны үнемдеу үшін JSONL немесе TXT файлдарынан мәтінді ағындық оқу"""
    batch = []
    for filepath in corpus_files:
        if not os.path.exists(filepath):
            continue
        print(f"  -> Токенизатор үшін дерек оқылуда: {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if filepath.endswith(".jsonl"):
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                    except Exception:
                        text = line
                else:
                    text = line
                
                if len(text) > 10:
                    batch.append(text)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch

def train_kazakh_tokenizer(corpus_files, output_dir="tokenizer/vocab", vocab_size=16384):
    print("=" * 80)
    print(" YABGU ҚАЗАҚ ТІЛДІК ТОКЕНИЗАТОРЫН НӨЛДЕН ОҚЫТУ")
    print(f" Архитектура: Byte-level BPE | Сөздік көлемі: {vocab_size:,} токен")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. BPE моделінің негізін құру
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    
    # 2. ByteLevel Pre-tokenizer: бос орындар мен байттарды алдын-ала сақтайды (GPT-2/LLaMA стилі)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # 3. Декодер: байттарды кері мәтінге қайтарады
    tokenizer.decoder = ByteLevelDecoder()
    
    # 4. Тренерді баптау
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )
    
    # 5. Оқыту процесі
    print("[1/3] Корпус бойынша жиіліктер мен бірігулерді (merges) үйрету...")
    tokenizer.train_from_iterator(text_iterator(corpus_files), trainer=trainer)
    
    # 6. Post-processor: автоматты түрде <|bos|> және <|eos|> қосу
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|sep|> $B <|eos|>",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
            ("<|sep|>", tokenizer.token_to_id("<|sep|>")),
        ],
    )
    
    # 7. Сақтау
    tokenizer_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)
    print(f"[2/3] Токенизатор сәтті сақталды: {tokenizer_file}")
    
    # 8. Қазақ тіліне тексеру (Verification Test)
    print("\n[3/3] ТОКЕНИЗАТОРДЫ ҚАЗАҚ ТІЛІНЕ ТЕКСЕРУ:")
    test_sentences = [
        "Қазақстанның елордасы — Астана қаласы.",
        "Абайдың «Қара сөздері» терең пәлсапалық ойларға толы.",
        "Балалар мектептен үйлеріне қуана-қуана оралды.",
        "Әділет, білім, өнер — ел тірегі.",
    ]
    
    for s in test_sentences:
        encoded = tokenizer.encode(s)
        tokens = encoded.tokens
        ids = encoded.ids
        decoded = tokenizer.decode(ids)
        print(f"\nСөйлем: '{s}'")
        print(f"Токендер саны: {len(tokens)}")
        print(f"Токендер: {tokens}")
        print(f"ID-лер: {ids[:8]}... (жалпы {len(ids)})")
        print(f"Қайта қалпына келуі: '{decoded}'")
        assert decoded.strip() == s.strip(), "Декодтау сәйкес келмеді!"

    print("\n" + "=" * 80)
    print(" ҚАЗАҚША ТОКЕНИЗАТОР СӘТТІ ҮЙРЕТІЛДІ ЖӘНЕ ДАЙЫН!")
    print("=" * 80)
    return tokenizer

if __name__ == "__main__":
    # Бастапқы тексеру үшін тазартылған деректер
    data_files = glob.glob("data/processed/*.jsonl") + glob.glob("data/raw/*.jsonl")
    if not data_files:
        print("[ЕСКЕРТУ] Бірде-бір дерек файлы табылмады. Алдымен 'data/raw' немесе 'data/processed' ішіне дерек жинаңыз.")
    else:
        train_kazakh_tokenizer(data_files, vocab_size=16384)
