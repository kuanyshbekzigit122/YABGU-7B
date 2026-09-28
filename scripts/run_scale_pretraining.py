"""
YABGU Pretraining Scaling Pipeline (Direction 2: 50M+ Tokens)
1. Ingest 50,000+ quality Kazakh documents (Wikipedia KK + mC4 KK + ALASH Books);
2. Tokenize with existing 16K BPE tokenizer and pack into uint16 memmap;
3. Initialize from pre-trained YABGU-50M weights;
4. Pretrain for 3,000 steps (~49.1M tokens processed) on NVIDIA Tesla T4 GPU;
5. Run comprehensive linguistic evaluation.
"""

import sys
import os
import time
import json
import re
import math
import unicodedata
import subprocess
import numpy as np

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Root paths
for p in ["/content", "."]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    if os.path.exists("/content"):
        os.chdir("/content")
except Exception:
    pass

import torch
from tokenizers import Tokenizer
from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer
from training.dataloader import MemmapDataset
from scripts.generate import generate_text

print("=" * 80)
print(" YABGU-50M PRETRAINING SCALING: 50M+ ТОКЕНДІК ҮЛКЕН КОРПУС")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ==============================================================================
# 1. ДЕРЕКТЕРДІ АҒЫНДЫҚ ЖИНАУ ЖӘНЕ ТАЗАРТУ
# ==============================================================================
os.makedirs("/content/data/processed", exist_ok=True)
corpus_file = "/content/data/processed/corpus_50m.jsonl"

def clean_kz_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[←\d+\][^\n]*", "", text)
    text = re.sub(r"\b\d+\s+Әдеби\s+KZ\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # І/і гомоглифтерін түзету
    text = re.sub(r"(?<=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])[iI](?=[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])", "і", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# Егер үлкен корпус әлі жиналмаған болса:
if not os.path.exists(corpus_file) or os.path.getsize(corpus_file) < 10 * 1024 * 1024:
    print("\n[1/4] ҚАЗАҚША ДЕРЕКТЕРДІ АҒЫНДЫҚ ЖИНАУ (Wikipedia + mC4 + ALASH)...")
    from datasets import load_dataset

    doc_count = 0
    total_chars = 0

    with open(corpus_file, "w", encoding="utf-8") as f_out:
        # A) ALASH кітаптары
        alash_path = "/content/data/processed/alash_clean.jsonl"
        if os.path.exists(alash_path):
            with open(alash_path, "r", encoding="utf-8") as f_alash:
                for line in f_alash:
                    if line.strip():
                        f_out.write(line.strip() + "\n")
                        doc_count += 1
                        data = json.loads(line)
                        total_chars += len(data.get("text", ""))
            print(f"  -> [ALASH] {doc_count} кітап қосылды.")

        # B) Уикипедия (25,000 сапалы мақала)
        target_wiki = 25000
        print(f"  -> [WIKI] {target_wiki:,} мақала жүктелуде...")
        try:
            ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
            wiki_cnt = 0
            for item in ds_wiki:
                raw_text = item.get("text", "")
                title = item.get("title", "")
                if len(raw_text) < 150 or "айрық" in title.lower():
                    continue
                cleaned = clean_kz_text(raw_text)
                doc = {
                    "id": f"wiki_{wiki_cnt+1:06d}",
                    "title": title,
                    "text": cleaned,
                    "source": "wikipedia_kk",
                    "tokens_est": len(cleaned) // 5
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                doc_count += 1
                wiki_cnt += 1
                total_chars += len(cleaned)
                if wiki_cnt % 5000 == 0:
                    print(f"     Уикипедия: {wiki_cnt:,}/{target_wiki:,} мақала | Таңба: ~{total_chars:,}")
                if wiki_cnt >= target_wiki:
                    break
        except Exception as e:
            print(f"  [ЕСКЕРТУ] Уикипедия жүктеу: {e}")

        # C) mC4 Kazakh (25,000 сапалы мәтін: жаңалықтар, ғылым, мақалалар)
        target_c4 = 25000
        print(f"  -> [mC4] {target_c4:,} сапалы қазақша веб/ғылыми құжат жүктелуде...")
        try:
            ds_c4 = load_dataset("allenai/c4", "kk", split="train", streaming=True)
            c4_cnt = 0
            for item in ds_c4:
                raw_text = item.get("text", "")
                if len(raw_text) < 200 or len(raw_text) > 30000:
                    continue
                cleaned = clean_kz_text(raw_text)
                # Қазақша әріптер үлесін тексеру (ластанған мәтіндерді өткізбеу)
                kz_specific = sum(1 for c in cleaned if c in "әғқңөұүһіӘҒҚҢӨҰҮҺІ")
                if kz_specific < 3:
                    continue
                
                doc = {
                    "id": f"mc4_{c4_cnt+1:06d}",
                    "title": "",
                    "text": cleaned,
                    "source": "mc4_kk",
                    "tokens_est": len(cleaned) // 5
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                doc_count += 1
                c4_cnt += 1
                total_chars += len(cleaned)
                if c4_cnt % 5000 == 0:
                    print(f"     mC4: {c4_cnt:,}/{target_c4:,} құжат | Таңба: ~{total_chars:,}")
                if c4_cnt >= target_c4:
                    break
        except Exception as e:
            print(f"  [ЕСКЕРТУ] mC4 жүктеу: {e}")

    size_mb = os.path.getsize(corpus_file) / (1024 * 1024)
    print(f"[OK] Үлкен корпус дайын: {doc_count:,} құжат | Көлемі: {size_mb:.1f} MB | Шамамен токен: ~{total_chars//5:,}")
else:
    print(f"[1/4] Корпус бұрын дайындалған: {corpus_file} ({os.path.getsize(corpus_file)/(1024*1024):.1f} MB)")

# ==============================================================================
# 2. МЕММАП БИНАРЛЫҚ ЖИНАҚТАУ (train_50m.bin, val_50m.bin)
# ==============================================================================
os.makedirs("/content/data/tokenized", exist_ok=True)
train_bin = "/content/data/tokenized/train_50m.bin"
val_bin = "/content/data/tokenized/val_50m.bin"

if not os.path.exists(train_bin) or os.path.getsize(train_bin) < 5 * 1024 * 1024:
    print("\n[2/4] ДЕРЕКТЕРДІ 16K BPE ТОКЕНИЗАТОРЫМЕН БИНАРЛЫҚ МЕММАПҚА ЖИНАУ...")
    tok_path = "/content/tokenizer/vocab/tokenizer.json"
    tokenizer = Tokenizer.from_file(tok_path)
    all_tokens = []
    t0 = time.time()

    with open(corpus_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "")
            enc = tokenizer.encode(text)
            all_tokens.extend(enc.ids)
            if (idx + 1) % 10000 == 0:
                print(f"  -> Токенизация: {idx+1:,} құжат | Токендер: {len(all_tokens):,}")

    total_tokens = len(all_tokens)
    print(f"[OK] Жалпы жиналған бинарлық токендер: {total_tokens:,} ({time.time()-t0:.1f} сек)")

    val_len = int(total_tokens * 0.05)
    train_arr = np.array(all_tokens[:-val_len], dtype=np.uint16)
    val_arr = np.array(all_tokens[-val_len:], dtype=np.uint16)

    train_arr.tofile(train_bin)
    val_arr.tofile(val_bin)
    print(f"  Train: {len(train_arr):,} токен ({os.path.getsize(train_bin)/(1024*1024):.1f} MB)")
    print(f"  Val:   {len(val_arr):,} токен ({os.path.getsize(val_bin)/(1024*1024):.1f} MB)")
else:
    print(f"[2/4] Бинарлық деректер бұрын жиналған: {train_bin}")

# ==============================================================================
# 3. YABGU-50M PRETRAINING (3,000 ҚАДАМ, ~49M ТОКЕН)
# ==============================================================================
print("\n[3/4] PRETRAINING: 3,000 ҚАДАМДЫҚ ТЕРЕҢ ОҚЫТУ БАСТАЛДЫ...")

cfg = YabguConfig(
    vocab_size=16384,
    max_seq_len=256,
    dim=512,
    n_layers=12,
    n_heads=8,
    intermediate_size=1408,
    micro_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=4e-4,
    min_lr=4e-5,
    warmup_steps=150,
    weight_decay=0.1,
    mixed_precision="fp16"
)

# Модельді құру
model = YabguTransformer(cfg).to(device)

# Pretrained салмақтарды жүктеу (Continuous Pretraining)
fp16_init = "/content/checkpoints/yabgu_50m_fp16.pt"
last_init = "/content/checkpoints/checkpoint_last.pt"
init_path = None
for cp in [fp16_init, last_init]:
    if os.path.exists(cp):
        init_path = cp
        break

if init_path:
    print(f"[CONTINUOUS PRETRAINING] Базалық модель салмақтары жүктелуде: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    state = {k: v.to(torch.float32) for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    print("[CONTINUOUS PRETRAINING] Базалық грамматика сақталды, жаңа білім сіңірілуде!")
else:
    print("[PRETRAINING] Нөлден оқыту басталуда.")

# DataLoader
train_ds = MemmapDataset(train_bin, seq_len=cfg.max_seq_len)
val_ds = MemmapDataset(val_bin, seq_len=cfg.max_seq_len)

# Optimizer & Scaler
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {"params": decay_params, "weight_decay": cfg.weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

def get_lr(step, max_steps):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step > max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

@torch.no_grad()
def estimate_loss(iters=30):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = val_ds.get_batch(cfg.micro_batch_size, device=device)
        with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(x, targets=y)
        losses[k] = loss.item()
    model.train()
    vl = losses.mean().item()
    ppl = math.exp(min(vl, 20))
    return vl, ppl

max_steps = 3000
eval_interval = 150
save_interval = 500
ckpt_dir = "/content/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

print("=" * 80)
print(f"ЖАТТЫҒУ ЦИКЛІ БАСТАЛДЫ | 0 -> {max_steps} қадам | Эффективті batch: {cfg.micro_batch_size * cfg.gradient_accumulation_steps}")
print("=" * 80)

t0 = time.time()
for step in range(max_steps + 1):
    lr = get_lr(step, max_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Эвалюация
    if step % eval_interval == 0 or step == max_steps:
        vl, vppl = estimate_loss()
        print(f"\n>>> [STEP {step:05d}/{max_steps:05d}] VAL LOSS: {vl:.4f} | VAL PPL: {vppl:.2f} | LR: {lr:.2e} <<<")

    # Чекпоинт сақтау
    if (step > 0 and step % save_interval == 0) or step == max_steps:
        save_data = {
            "step": step,
            "model": model.state_dict(),
            "config": cfg,
            "val_loss": vl if 'vl' in locals() else None,
            "val_ppl": vppl if 'vppl' in locals() else None
        }
        torch.save(save_data, f"{ckpt_dir}/checkpoint_scale_last.pt")
        if step % 1000 == 0 or step == max_steps:
            torch.save(save_data, f"{ckpt_dir}/checkpoint_scale_step_{step:05d}.pt")
        print(f"[CHECKPOINT] Сақталды: checkpoint_scale_last.pt (Step {step})")

    if step == max_steps:
        break

    # Тренинг қадамы (Gradient accumulation)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for micro_step in range(cfg.gradient_accumulation_steps):
        x, y = train_ds.get_batch(cfg.micro_batch_size, device=device)
        with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(x, targets=y)
            loss = loss / cfg.gradient_accumulation_steps
        accum_loss += loss.item()
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    if step % 50 == 0 and step > 0:
        elapsed = time.time() - t0
        tok_speed = (step * cfg.micro_batch_size * cfg.gradient_accumulation_steps * cfg.max_seq_len) / max(elapsed, 1e-5)
        print(f"Step {step:05d} | Train Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_speed:.0f} tok/s")

# ==============================================================================
# 4. ТІЛДІК БАҒАЛАУ ЖӘНЕ ГЕНЕРАЦИЯ
# ==============================================================================
print("\n" + "=" * 80)
print(" ТЕРЕҢ ОҚЫТЫЛҒАН YABGU-50M МОДЕЛІН БАҒАЛАУ (INFERENCE)")
print("=" * 80)

eval_prompts = [
    "Қазақстанның экономикалық даму бағыттары — ",
    "Қазақ тілінің мемлекеттік мәртебесі ",
    "Абай Құнанбайұлының философиялық көзқарасы ",
    "Ғылым мен жасанды интеллект технологиялары ",
    "Қазақстанның табиғи байлықтары мен географиясы "
]

last_scale_ckpt = f"{ckpt_dir}/checkpoint_scale_last.pt"
tok_path = "/content/tokenizer/vocab/tokenizer.json"

for p in eval_prompts:
    print(f"\n--- ПРОМПТ: '{p}' ---")
    generate_text(
        prompt=p,
        checkpoint_path=last_scale_ckpt,
        tokenizer_path=tok_path,
        max_new_tokens=60,
        temperature=0.7,
        top_k=40
    )

print("\n" + "=" * 80)
print(" PRETRAINING SCALING ТОЛЫҚ АЯҚТАЛДЫ! ")
print("=" * 80)
