import os
import zipfile

zip_path = "/content/yabgu_pkg.zip"
if os.path.exists(zip_path):
    print(f"Unzipping {zip_path} to /content ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/content")
    print("Unzip completed successfully.")
else:
    print(f"Warning: {zip_path} not found!")

# Verification
required = [
    "/content/models/transformer.py",
    "/content/training/sft_trainer.py",
    "/content/training/sft_dataset.py",
    "/content/scripts/run_colab_sft.py",
    "/content/data/sft/kazakh_sft_combined.jsonl",
    "/content/tokenizer/vocab/tokenizer.json"
]

print("\n--- Files verification ---")
all_ok = True
for r in required:
    exists = os.path.exists(r)
    sz = os.path.getsize(r) if exists else 0
    print(f"[{'OK' if exists else 'MISSING'}] {r} ({sz:,} bytes)")
    if not exists:
        all_ok = False

ckpt_step_3000 = "/content/checkpoints/checkpoint_step_3000.pt"
ckpt_scale3000 = "/content/checkpoints/yabgu_50m_scale3000_fp16.pt"
print(f"Checkpoint step 3000 exists: {os.path.exists(ckpt_step_3000)}")
print(f"Checkpoint scale3000 fp16 exists: {os.path.exists(ckpt_scale3000)}")

if all_ok:
    print("\n>>> ALL SFT REQUIREMENTS VERIFIED! READY TO LAUNCH SFT TRAINING! <<<")
