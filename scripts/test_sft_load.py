import sys
import os
import importlib

# Force clean reload from disk
for mod in list(sys.modules.keys()):
    if mod.startswith(("models", "training", "configs")):
        del sys.modules[mod]

sys.path.insert(0, "/content")
from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer
import torch

config = YabguConfig()
model = YabguTransformer(config)
ckpt_path = "/content/checkpoints/yabgu_50m_scale3000_fp16.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if "model" in ckpt:
    state_dict = ckpt["model"]
elif "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    state_dict = ckpt

# Float conversion if fp16
state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
print(f"Successfully loaded checkpoint with {model.count_parameters():,} parameters!")

# Test forward pass with prompt masking
x = torch.randint(0, 1000, (2, 64))
y = x.clone()
y[:, :20] = -100 # Mask first 20 tokens
logits, loss = model(x, targets=y, ignore_index=-100)
print(f"Forward pass OK! Loss: {loss.item():.4f}")
