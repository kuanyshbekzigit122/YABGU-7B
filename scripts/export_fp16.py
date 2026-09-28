import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from configs.yabgu_50m import YabguConfig

src_ckpt = "checkpoints/checkpoint_colab_1500.pt"
out_fp16 = "checkpoints/yabgu_50m_fp16.pt"

print(f"Loading {src_ckpt}...")
ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
model_state = ckpt["model"]

# Convert weights to fp16
fp16_state = {}
for k, v in model_state.items():
    if torch.is_floating_point(v):
        fp16_state[k] = v.to(torch.float16)
    else:
        fp16_state[k] = v

export_data = {
    "model": fp16_state,
    "config": ckpt.get("config", YabguConfig()),
    "step": ckpt.get("step", 1500),
    "val_loss": 4.21,
    "val_ppl": 67.99,
    "framework": "PyTorch",
    "architecture": "YABGU-50M (SwiGLU + RoPE + RMSNorm)"
}

torch.save(export_data, out_fp16)
size_mb = os.path.getsize(out_fp16) / (1024 * 1024)
print(f"[OK] Exported {out_fp16} successfully! Size: {size_mb:.2f} MB")
