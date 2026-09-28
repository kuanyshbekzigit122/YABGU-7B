# 🇰🇿 YABGU: Қазақ Тіліндегі Дербес Тілдік Модель (From-Scratch Kazakh LLM)

**YABGU** — қазақ тілінің бай лексикалық қорын, агглютинативті грамматикасын және сингармонистік табиғатын негізге ала отырып, **нөлден бастап жобаланған және оқытылған дербес үлкен тілдік модель (LLM)**.

Бұл жоба сыртқы дайын API (OpenAI, Anthropic) немесе шетелдік модельдерді (LLaMA, Mistral) жай ғана қазақшаға аударып баптау (fine-tuning) емес. Мұнда архитектура, токенизатор, деректер құбыры, алдын ала оқыту (Pretraining) және нұсқаулықтық баптау (Instruction SFT) толығымен дербес жүзеге асырылды.

---

## 📌 Архитектуралық Ерекшеліктері (YABGU-50M)

| Параметр | Мәні | Сипаттамасы |
| :--- | :--- | :--- |
| **Параметр саны** | **46,936,576 (~46.94M)** | Есептелген таза салмақтар саны |
| **Қабаттар саны (Layers)** | **12** | Decoder-only Transformer блогы |
| **Жасырын өлшем ($d_{model}$)** | **512** | Hidden dimension |
| **Attention Heads** | **8** | Әр head өлшемі: 64 ($512 / 8$) |
| **FFN Архитектурасы** | **SwiGLU (1408)** | LLaMA/Mistral стиліндегі белсендіру |
| **Позициялық кодтау** | **RoPE (Rotary Position Embeddings)** | Ұзын мәтіндерге тұрақты контекст |
| **Нормализация** | **RMSNorm ($\epsilon=10^{-5}$)** | Жылдам әрі тұрақты градиент |
| **Сөздік көлемі (Vocab)** | **16,384 токен** | Byte-level BPE, 100% қазақша қамту, 0% OOV |
| **Салмақтарды байлау (Weight Tying)** | **Қосылған** | Input Embedding пен LM Head ортақ салмақта |

---

## 📊 Оқыту Нәтижелері (Tesla T4 16GB GPU)

### 1. 2-Бағыт: Ауқымды Pretraining (Foundation Model)
- **Деректер қоры:** 50,004 тазартылған қазақша құжат (253.7 МБ, Уикипедия + mC4 ғылыми мақалалары + Алаш әдеби классикасы).
- **Токендер саны:** **35,923,337 токен** (Train) және **1,890,701 токен** (Validation).
- **Оқыту қадамы:** 3,000 қадам (~49,152,000 токен өңделді, ~28,700 токен/сек).
- **Validation Loss:** `9.80` $\rightarrow$ **`3.78`**
- **Perplexity (PPL):** `18,008` $\rightarrow$ **`44.11`**
- **Чекпоинт:** `checkpoints/yabgu_50m_scale3000_fp16.pt`

### 2. 1-Бағыт: Instruction Fine-Tuning (SFT / AI Assistant)
- **Нұсқаулықтар базасы:** **60,967 сапалы қазақша нұсқаулық пен диалог** (AmanMussa Alpaca-v2, DarkyMan Kaz-Dialogues, төл тарих пен тұлғалар).
- **Prompt Masking:** Модель тек пайдаланушы сұрағына берілетін Assistant жауаптары бойынша ғана loss есептеп үйренді.
- **Оқыту қадамы:** 1,000 қадам (Cosine LR Schedule: `1.2e-4` $\rightarrow$ `1.0e-5`).
- **Validation Loss:** `3.78` $\rightarrow$ **`3.1384`**
- **Perplexity (PPL):** `44.11` $\rightarrow$ **`23.07`**
- **Чекпоинт:** `checkpoints/yabgu_50m_instruct_fp16.pt`

---

## 🚀 Жобаны Іске Қосу және Қолдану

### 1. Тәуелділіктерді орнату:
```bash
pip install -r requirements.txt
```

### 2. Интерактивті Қазақша AI Чат Көмекшісі (SFT Instruct режимі):
```bash
python scripts/interactive_chat.py
```
> Мысал:
> ```
> Сіз: Қазақстанның астанасы қандай қала?
> YABGU: Қазақстанның астанасы — Астана қаласы.
> ```

### 3. Мәтін жалғастыру (Base Foundation режимі):
```bash
python scripts/generate.py "Қазақстанның бай табиғаты "
```

---

## 📂 Жоба Құрылымы

```
YABGU-7B/
├── configs/              # Модель конфигурациялары (yabgu_50m.py, yabgu_350m.py)
├── models/               # PyTorch Transformer, RoPE, RMSNorm, SwiGLU
├── tokenizer/            # 16K Byte-level BPE қазақша токенизатор
├── training/             # Memmap DataLoader, Pretraining & SFT Trainer
├── checkpoints/          # Оқытылған модель салмақтары (Git LFS)
│   ├── yabgu_50m_scale3000_fp16.pt    # Pretrained Foundation моделі
│   └── yabgu_50m_instruct_fp16.pt     # SFT Интерактивті Чат көмекшісі
├── scripts/              # Интерактивті чат, генерация, Colab оқыту құралдары
└── README.md             # Жоба сипаттамасы мен құжаттамасы
```

---

## 🗺️ Roadmap (Масштабтау Жоспары)
- [x] **1-Кезең: YABGU-50M Foundation**: Нөлден токенизация, RoPE+SwiGLU архитектурасы (Орындалды).
- [x] **2-Кезең: Data Scaling**: 50,004 құжат, 38M токен, Val PPL 44.11 (Орындалды).
- [x] **3-Кезең: Instruction Fine-Tuning (SFT)**: 60,967 диалог, Val PPL 23.07, қазақша AI көмекші режимі (Орындалды).
- [ ] **4-Кезең: YABGU-350M**: 24 қабат, $d_{model}=1024$, GQA қолдайтын 100M+ токендік кеңейтілген модель.
- [ ] **5-Кезең: YABGU-1B / 7B**: Заманауи ұлттық өндірістік деңгейдегі модель.

---
**Автор:** Бекжігіт Қуаныш ([kuanyshbekzigit122/YABGU-7B](https://github.com/kuanyshbekzigit122/YABGU-7B.git))  
**Лицензия:** Apache 2.0 / MIT
