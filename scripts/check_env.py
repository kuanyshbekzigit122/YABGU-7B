"""
YABGU Project - STEP 1: Environment Diagnostic & System Verification Script
Бұл скрипт жергілікті ортаны және Google Colab (NVIDIA T4 16GB) ортасын тексереді.
"""

import sys
import os
import platform
import shutil
import importlib

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def check_python():
    print("=" * 60)
    print(" 1. PYTHON ЖӘНЕ ОПЕРАЦИЯЛЫҚ ЖҮЙЕ ТЕКСЕРІСІ")
    print("=" * 60)
    py_version = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    print(f"[OK] Python нұсқасы: {py_version}")
    print(f"[OK] ОЖ (OS): {os_info}")
    
    if sys.version_info < (3, 9):
        print("[ЕСКЕРТУ] Python нұсқасы 3.9-дан төмен. Жоба үшін 3.9+ немесе 3.10+ ұсынылады.")
    else:
        print("[OK] Python нұсқасы сәйкес келеді.")

def check_pytorch_and_cuda():
    print("\n" + "=" * 60)
    print(" 2. PYTORCH ЖӘНЕ GPU / CUDA ТЕКСЕРІСІ")
    print("=" * 60)
    try:
        import torch
        print(f"[OK] PyTorch нұсқасы: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA қолжетімді ме?: {'ИӘ (GPU бар)' if cuda_available else 'ЖОҚ (Тек CPU режимі)'}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"[OK] Қолжетімді GPU саны: {device_count}")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  - GPU {i}: {device_name}")
                print(f"    Жалпы VRAM: {total_memory:.2f} GB")
                
                # Check for Tesla T4 or similar
                if "T4" in device_name:
                    print("    -> [СӘЙКЕС] Google Colab NVIDIA Tesla T4 анықталды!")
            
            # Simple CUDA tensor test
            x = torch.ones((2, 2), device="cuda")
            print("[OK] GPU-да қарапайым тензор операциясы сәтті орындалды.")
        else:
            print("[МАҢЫЗДЫ ЕСКЕРТПЕ] Қазір тек CPU анықталды. Жергілікті ортада деректерді өңдеуге")
            print("болады, ал YABGU-50M моделін оқыту Google Colab (GPU: T4 16GB)-та жүргізіледі.")
    except ImportError:
        print("[ҚАТЕ] PyTorch орнатылмаған! 'pip install torch' орындаңыз.")

def check_packages():
    print("\n" + "=" * 60)
    print(" 3. МАҢЫЗДЫ КІТАПХАНАЛАРДЫ ТЕКСЕРУ")
    print("=" * 60)
    required_libs = [
        ("numpy", "NumPy (тензорлық математика)"),
        ("tokenizers", "HuggingFace Tokenizers (жылдам Rust токенизаторы)"),
        ("sentencepiece", "SentencePiece (BPE/Unigram токенизациясы)"),
        ("pyarrow", "PyArrow (Parquet деректер сақтау)"),
        ("tqdm", "TQDM (progress bar)"),
        ("fasttext", "FastText (тілді анықтау фильтрі)"),
    ]
    
    for lib, desc in required_libs:
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "орнатылған")
            print(f"[OK] {lib:<15} ({desc}) -> Нұсқа: {ver}")
        except ImportError:
            print(f"[ЖОҚ] {lib:<15} ({desc}) -> ОРНАТЫЛМАҒАН (pip install {lib})")

def check_resources():
    print("\n" + "=" * 60)
    print(" 4. ДИСК ЖӘНЕ ЖАДЫ (RAM) РЕСУРСТАРЫ")
    print("=" * 60)
    total, used, free = shutil.disk_usage(".")
    print(f"Диск кеңістігі: Жалпы {total / (1024**3):.1f} GB | Бос орын: {free / (1024**3):.1f} GB")
    
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"Жедел жады (RAM): Жалпы {ram.total / (1024**3):.1f} GB | Бос: {ram.available / (1024**3):.1f} GB")
    except ImportError:
        print("Жедел жады көлемін толық көру үшін 'psutil' кітапханасын орнатыңыз.")
    
    print("\n" + "=" * 60)
    print(" ДИАГНОСТИКА АЯҚТАЛДЫ. YABGU жобасына қош келдіңіз!")
    print("=" * 60)

if __name__ == "__main__":
    check_python()
    check_pytorch_and_cuda()
    check_packages()
    check_resources()
