import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import HfApi

api = HfApi()
print("Searching HuggingFace for Kazakh instruction datasets...")
try:
    datasets = api.list_datasets(search="kazakh", limit=100)
    instruct_candidates = []
    for d in datasets:
        d_id = d.id.lower()
        if any(k in d_id for k in ["instruct", "alpaca", "dolly", "chat", "qa", "dialogue", "prompt"]):
            instruct_candidates.append(d.id)
            print(f"Candidate: {d.id} (downloads: {getattr(d, 'downloads', 'N/A')})")
    print(f"Total matching instruct candidates: {len(instruct_candidates)}")
except Exception as e:
    print(f"Error searching datasets: {e}")
