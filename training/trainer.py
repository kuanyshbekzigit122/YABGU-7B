"""
YABGU Pretraining Engine
Google Colab + NVIDIA Tesla T4 16GB VRAM ресурсына толықтай бейімделген жаттығу циклі.
Авто-checkpoint, mixed precision (fp16), gradient accumulation, cosine scheduler қолдайды.
"""

import sys
import os
import time
import math
import glob
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Project root-ты sys.path-қа қосу
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer
from training.dataloader import MemmapDataset

def get_lr(it: int, config: YabguConfig, max_iters: int) -> float:
    # 1. Linear warmup
    if it < config.warmup_steps:
        return config.learning_rate * it / config.warmup_steps
    # 2. Егер max_iters-тен асып кетсе
    if it > max_iters:
        return config.min_lr
    # 3. Cosine decay
    decay_ratio = (it - config.warmup_steps) / (max_iters - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def train(
    max_steps: int = 5000,
    eval_interval: int = 250,
    eval_iters: int = 40,
    save_interval: int = 500,
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
    config: YabguConfig = None
):
    print("=" * 80)
    print(" YABGU-50M PRETRAINING ENGINE ИСКЕ ҚОСЫЛУДА")
    print("=" * 80)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    if config is None:
        config = YabguConfig()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Жаттығу құрылғысы (Device): {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Dataset дайындығы
    train_bin = "data/tokenized/train.bin"
    val_bin = "data/tokenized/val.bin"
    
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("[ҚАТЕ] Бинарлық деректер табылмады. 'python scripts/prepare_dataset.py' орындаңыз!")
        return

    train_ds = MemmapDataset(train_bin, seq_len=config.max_seq_len)
    val_ds = MemmapDataset(val_bin, seq_len=config.max_seq_len)
    print(f"Train токендері: {train_ds.num_tokens:,} | Val токендері: {val_ds.num_tokens:,}")

    # 2. Модельді құру
    model = YabguTransformer(config).to(device)
    print(f"Модель параметрлері: {model.count_parameters():,} (~{model.count_parameters()/1e6:.2f}M)")

    # 3. Optimizer (AdamW)
    # Weight decay тек 2D салмақтарға қолданылады (эмбеддингтер мен RMSNorm бөлек)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Modern PyTorch 2.0+ AMP
    use_amp = (device == "cuda" and config.mixed_precision == "fp16")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # CPU-да жергілікті жылдам тексеру үшін бейімдеу
    effective_micro_batch = config.micro_batch_size if device == "cuda" else 2
    effective_accum = config.gradient_accumulation_steps if device == "cuda" else 2

    # 4. Checkpoint-тен жалғастыру
    start_step = 0
    last_ckpt = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    if resume and os.path.exists(last_ckpt):
        print(f"[RESUME] Соңғы checkpoint табылды: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"[RESUME] Жаттығу {start_step}-қадамнан жалғасуда.")

    # 5. Validation функциясы
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = val_ds.get_batch(effective_micro_batch, device=device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, targets=y)
            losses[k] = loss.item()
        model.train()
        val_loss = losses.mean().item()
        ppl = math.exp(min(val_loss, 20))  # Perplexity
        return val_loss, ppl

    # 6. Training циклі
    print("\n" + "=" * 80)
    print(f"PRETRAINING БАСТАЛДЫ | Қадамдар: {start_step} -> {max_steps} | Эффективті Batch: {config.micro_batch_size * config.gradient_accumulation_steps}")
    print("=" * 80)

    t0 = time.time()
    for step in range(start_step, max_steps + 1):
        # LR жаңарту
        lr = get_lr(step, config, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Эвалюация
        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_ppl = estimate_loss()
            print(f"\n>>> [STEP {step:05d}/{max_steps:05d}] VAL LOSS: {val_loss:.4f} | VAL PERPLEXITY: {val_ppl:.2f} | LR: {lr:.2e} <<<")

        # Checkpoint сақтау
        if (step > 0 and step % save_interval == 0) or step == max_steps:
            ckpt_data = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "config": config,
            }
            torch.save(ckpt_data, last_ckpt)
            step_ckpt = os.path.join(checkpoint_dir, f"checkpoint_step_{step:05d}.pt")
            torch.save(ckpt_data, step_ckpt)
            print(f"[CHECKPOINT] Сақталды: {last_ckpt} және {step_ckpt}")

        if step == max_steps:
            break

        # Forward & Backward (Gradient Accumulation)
        optimizer.zero_grad(set_to_none=True)
        micro_loss_accum = 0.0

        for micro_step in range(effective_accum):
            x, y = train_ds.get_batch(effective_micro_batch, device=device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, targets=y)
                loss = loss / effective_accum
            micro_loss_accum += loss.item()
            scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        # Прогресс бақылау
        if step % 25 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (config.micro_batch_size * config.gradient_accumulation_steps * config.max_seq_len * 25) / max(dt, 1e-5)
            print(f"Step {step:05d} | Train Loss: {micro_loss_accum:.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")

    print("\n" + "=" * 80)
    print(" PRETRAINING СӘТТІ АЯҚТАЛДЫ! МОДЕЛЬ САҚТАЛДЫ.")
    print("=" * 80)

if __name__ == "__main__":
    train(max_steps=100, eval_interval=50, save_interval=50)
