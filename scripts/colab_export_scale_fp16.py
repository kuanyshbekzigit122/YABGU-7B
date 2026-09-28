import os
import torch

src = "/content/checkpoints/checkpoint_scale_last.pt"
dst = "/content/checkpoints/yabgu_50m_scale3000_fp16.pt"

print(f"Loading {src} ({os.path.getsize(src)/(1024*1024):.1f} MB)...")
ckpt = torch.load(src, map_location="cpu", weights_only=False)

fp16_model = {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in ckpt["model"].items()}
export = {
    "step": ckpt.get("step", 3000),
    "model": fp16_model,
    "config": ckpt.get("config"),
    "val_loss": ckpt.get("val_loss", 3.78),
    "val_ppl": ckpt.get("val_ppl", 44.11)
}
torch.save(export, dst)
print(f"[OK] Exported fp16 checkpoint: {dst} ({os.path.getsize(dst)/(1024*1024):.1f} MB)")
