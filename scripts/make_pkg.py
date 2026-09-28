import zipfile
import os

zip_name = "yabgu_pkg.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ["configs", "models", "training", "tokenizer", "scripts"]:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith((".py", ".json", ".md")) and not f.startswith("."):
                    filepath = os.path.join(root, f)
                    arcname = os.path.relpath(filepath, ".")
                    z.write(filepath, arcname)
    if os.path.exists("data/processed/alash_clean.jsonl"):
        z.write("data/processed/alash_clean.jsonl", "data/processed/alash_clean.jsonl")

print(f"[OK] {zip_name} жасалды: {os.path.getsize(zip_name)/(1024*1024):.2f} MB")
