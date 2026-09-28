import zipfile
import os
import sys

zip_path = "/yabgu_pkg.zip" if os.path.exists("/yabgu_pkg.zip") else "yabgu_pkg.zip"
print(f"[BOOTSTRAP] Unzipping {zip_path} into /content...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/content")

print("[BOOTSTRAP] Files extracted to /content:")
for item in os.listdir("/content"):
    print(" -", item)

print("[BOOTSTRAP] Checking Python imports...")
sys.path.insert(0, "/content")
from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer
import torch

cfg = YabguConfig()
model = YabguTransformer(cfg)
print(f"[BOOTSTRAP] Model successfully initialized: {model.count_parameters():,} parameters!")
print(f"[BOOTSTRAP] CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
