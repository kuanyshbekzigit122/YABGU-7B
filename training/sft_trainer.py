"""
YABGU Instruction Fine-Tuning (SFT) Engine
Табиғи қазақ тілінде сұрақ-жауап, нұсқаулықтарды орындау және диалог режиміне бейімдеу.
Алдын ала оқытылған базалық салмақты (Foundation Model) жүктеп, Assistant жауаптары бойынша ғана loss есептейді (Prompt Masking).
"""

import sys
import os
import time
import math
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer

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
from training.sft_dataset import KazakhSFTDataset

def get_cosine_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_steps:
        return max_lr * (it + 1) / (warmup_steps + 1)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast(dtype=torch.float16 if device == "cuda" else torch.float32):
                _, loss = model(input_ids, targets=labels, ignore_index=-100)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                total_batches += 1
            if total_batches >= 40:
                break
    model.train()
    if total_batches == 0:
        return 999.0, 999.0
    avg_loss = total_loss / total_batches
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl

def train_sft(
    base_checkpoint: str,
    data_path: str,
    tokenizer_path: str = "tokenizer/vocab/tokenizer.json",
    output_dir: str = "checkpoints",
    epochs: int = 2,
    max_steps: int = 1000,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    learning_rate: float = 1.2e-4,
    min_lr: float = 1.0e-5,
    warmup_steps: int = 50,
    max_seq_len: int = 512,
    eval_interval: int = 50,
    save_interval: int = 100,
):
    print("=" * 80)
    print(" YABGU-50M INSTRUCTION SFT (SUPERVISED FINE-TUNING) ИСКЕ ҚОСЫЛУДА")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HW] Қолданылатын құрылғы: {device.upper()}")
    if device == "cuda":
        print(f"[HW] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[HW] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    config = YabguConfig(max_seq_len=max_seq_len)
    model = YabguTransformer(config)
    
    # Базалық алдын ала оқытылған салмақты жүктеу
    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Базалық чекпоинт табылмады: {base_checkpoint}")
        
    print(f"[LOAD] Базалық Foundation модель салмағы жүктелуде: {base_checkpoint}")
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    # Дәл сәйкестендіріп жүктеу (қажет болса float32-ге айналдыру)
    state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] Модель параметрі: {model.count_parameters():,} салмақ жүктелді.")
    model.to(device)
    model.train()
    
    # Датасетті дайындау
    print(f"[DATA] SFT нұсқаулықтар дерегі жүктелуде: {data_path}")
    full_dataset = KazakhSFTDataset(data_path, tokenizer_path, max_seq_len=max_seq_len)
    total_len = len(full_dataset)
    print(f"[DATA] Жалпы диалог/нұсқаулық жұптары: {total_len:,}")
    
    val_len = max(int(total_len * 0.05), 50)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[DATA] Train: {train_len:,} дана | Validation: {val_len:,} дана")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1 if sys.platform != "win32" else 0,
        pin_memory=(device == "cuda")
    )
    
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = max_steps if max_steps is not None else steps_per_epoch * epochs
    print(f"[TRAIN] Эпохалар: {epochs} | Қадамдар/эпоха: {steps_per_epoch} | Жалпы оқыту қадамы: {total_steps}")
    print(f"[TRAIN] Effective Batch Size: {batch_size * grad_accum_steps} (Micro: {batch_size}, Accum: {grad_accum_steps})")
    print(f"[TRAIN] Peak LR: {learning_rate} | Min LR: {min_lr} | Warmup: {warmup_steps} қадам")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )
    scaler = GradScaler(enabled=(device == "cuda"))
    
    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    
    print("-" * 80)
    print(" SFT БАПТАУ ЦИКЛІ БАСТАЛДЫ")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        print(f"\n>>> [EPOCH {epoch}/{epochs}] Басталды...")
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro_step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with autocast(dtype=torch.float16 if device == "cuda" else torch.float32):
                _, loss = model(input_ids, targets=labels, ignore_index=-100)
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps
            
            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Dynamic learning rate update
                curr_lr = get_cosine_lr(step, warmup_steps, total_steps, learning_rate, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                
                # Лог шығару
                if step % 20 == 0 or step == total_steps:
                    dt = time.time() - t0
                    tokens_per_sec = (batch_size * grad_accum_steps * max_seq_len * 20) / max(dt, 1e-4) if step > 20 else 0
                    train_ppl = math.exp(min(accum_loss, 20))
                    print(
                        f"[Step {step:04d}/{total_steps:04d}] "
                        f"Loss: {accum_loss:.4f} | "
                        f"PPL: {train_ppl:.2f} | "
                        f"LR: {curr_lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f}"
                    )
                    t0 = time.time()
                
                # Валидация
                if step % eval_interval == 0 or step == total_steps:
                    print(f"\n[EVAL] Step {step}: Валидация тексерілуде...")
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                    print(f"[EVAL] Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(output_dir, "yabgu_50m_instruct_best.pt")
                        torch.save({
                            "step": step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                            "config": config.__dict__,
                            "model_state_dict": model.state_dict()
                        }, best_path)
                        print(f"[*] Жаңа үздік модель сақталды: {best_path} (Val Loss: {val_loss:.4f})")
                    print("-" * 80)
                
                accum_loss = 0.0
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    # Финалдық экспорт: 16-bit жартылай дәлдіктегі шағын нұсқа
    final_fp16_path = os.path.join(output_dir, "yabgu_50m_instruct_fp16.pt")
    print(f"\n[EXPORT] Финалдық SFT моделі fp16 форматында сақталуда: {final_fp16_path}")
    half_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({
        "config": config.__dict__,
        "model_state_dict": half_state,
        "best_val_loss": best_val_loss,
        "type": "instruct_sft"
    }, final_fp16_path)
    print(f"[SUCCESS] SFT баптау сәтті аяқталды! Үздік Val Loss: {best_val_loss:.4f}")
    return final_fp16_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/yabgu_50m_scale3000_fp16.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/vocab/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()
    
    train_sft(
        base_checkpoint=args.base_checkpoint,
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        eval_interval=args.eval_interval
    )
