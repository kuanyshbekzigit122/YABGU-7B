import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from datasets import load_dataset

candidates = [
    "AmanMussa/kazakh-instruction-v2",
    "saillab/alpaca-kazakh-cleaned",
    "sabinaasker/kazakh_dolly",
    "DarkyMan/powerful-kazakh-dialogue"
]

for name in candidates:
    print(f"\n==========================================")
    print(f"Inspecting: {name}")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 3:
                break
            samples.append(item)
        print(f"Features: {list(samples[0].keys()) if samples else 'empty'}")
        for i, s in enumerate(samples):
            print(f"--- Sample {i+1} ---")
            for k, v in s.items():
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"  {k}: {val_str}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
