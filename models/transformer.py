"""
YABGU Decoder-Only Transformer Architecture
Архитектуралық ерекшеліктері:
- Rotary Position Embedding (RoPE)
- RMSNorm (LayerNorm орнына жылдам әрі тұрақты нормализация)
- SwiGLU Feed-Forward Networks
- Weight Tying (Input Embeddings пен Output LM Head салмақтарын байлау)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.yabgu_50m import YabguConfig
from models.rope import precompute_freqs_cis, apply_rotary_emb

class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    """SwiGLU Activation Function бар Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (silu(w1(x)) * w3(x)) @ w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with RoPE and Causal Masking"""
    def __init__(self, config: YabguConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        assert config.dim % config.n_heads == 0, "dim head санына бөлінуі тиіс"

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        # RoPE позициялық кодтауын қолдану
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose for Multi-head Attention: (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled Dot-Product Attention (PyTorch 2.0+ FlashAttention / SDPA)
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=(mask is None and seqlen > 1)
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with RMSNorm and SwiGLU"""
    def __init__(self, config: YabguConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config.dim, config.intermediate_size)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class YabguTransformer(nn.Module):
    """
    YABGU Causal Language Model
    Google Colab + NVIDIA T4 16GB үшін оңтайландырылған 10-50M толық Transformer архитектурасы.
    """
    def __init__(self, config: YabguConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight Tying (Эмбеддинг салмағын қайта пайдалану)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE freqs
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False
        )

        # Салмақтарды инициализациялау
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = input_ids.shape
        assert seqlen <= self.config.max_seq_len, f"Контекст шегінен асты: {seqlen} > {self.config.max_seq_len}"

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Causal Language Modeling Loss (Next Token Prediction)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=ignore_index
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Модельдің оқытылатын параметрлер санын есептеу"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Егер weight tying болса, ортақ параметрді екі рет санамау
        if self.config.tie_word_embeddings:
            tied_params = self.tok_embeddings.weight.numel()
            # PyTorch parameters() generator handles identical references, but check:
            unique_params = sum(p.numel() for p in set(self.parameters()) if p.requires_grad)
            return unique_params
        return total

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Авторегрессивті мәтін тудыру (Inference)"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Егер <|eos|> (id=2) кездессе, тоқтату
            if idx_next.item() == 2:
                break
        return idx
