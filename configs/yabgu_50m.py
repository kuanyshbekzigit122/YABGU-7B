"""
YABGU-50M Model Configuration
Бұл конфигурация Google Colab + NVIDIA Tesla T4 16GB VRAM ресурсына толықтай оңтайландырылған.
"""

from dataclasses import dataclass

@dataclass
class YabguConfig:
    # Model Architecture Parameters
    model_name: str = "YABGU-50M"
    vocab_size: int = 16384        # Қазақ тіліне оңтайланған сөздік көлемі (16K)
    max_seq_len: int = 1024        # Контекст ұзындығы (tokens)
    dim: int = 512                 # Hidden dimension (d_model)
    n_layers: int = 12             # Transformer қабаттарының саны
    n_heads: int = 8               # Self-attention heads саны (512 / 8 = 64 per head)
    n_kv_heads: int = 8            # Grouped Query Attention (GQA) қажет болса, әзірге MHA: 8
    intermediate_size: int = 1408  # SwiGLU FFN өлшемі (~8/3 * dim, 64-ке еселік)
    norm_eps: float = 1e-5         # RMSNorm epsilon
    rope_theta: float = 10000.0    # Rotary Position Embedding базалық мәні
    tie_word_embeddings: bool = True  # Input embedding пен LM Head салмақтарын байлау (параметр үнемдеу)
    dropout: float = 0.0           # Pretraining кезінде әдетте 0.0

    # Training & Colab T4 Optimizations
    micro_batch_size: int = 8      # T4 16GB VRAM-ға еркін сыятын batch
    gradient_accumulation_steps: int = 8  # Эффективті batch size = 8 * 8 = 64
    learning_rate: float = 6e-4    # AdamW үшін максималды LR
    min_lr: float = 6e-5           # Cosine decay төменгі шегі
    weight_decay: float = 0.1      # Регуляризация
    warmup_steps: int = 500        # Жылыну қадамдары
    max_grad_norm: float = 1.0     # Gradient clipping
    mixed_precision: str = "fp16"  # T4 Tensor Cores үшін ең жылдам әрі тұрақты формат
