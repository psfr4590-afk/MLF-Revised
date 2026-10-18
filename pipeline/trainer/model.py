"""Llama-style decoder-only transformer used by Model Lab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _ModuleBase = nn.Module
    _no_grad = torch.no_grad
except ImportError:  # Allow ModelConfig and sizing/provenance utilities to import without PyTorch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _ModuleBase = object
    def _no_grad():
        return lambda fn: fn


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_model: int = 768
    d_ffn: int = 2048
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        presets = {
            "117M": cls(n_layers=12, n_heads=12, n_kv_heads=12, d_model=768, d_ffn=2048),
            "360M": cls(n_layers=24, n_heads=16, n_kv_heads=16, d_model=1024, d_ffn=2816),
            "85M": cls(n_layers=10, n_heads=10, n_kv_heads=10, d_model=640, d_ffn=1728),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Options: {list(presets)}")
        return presets[name]

    @classmethod
    def from_target_params(
        cls,
        target_params: int,
        *,
        seq_len: int = 1024,
        vocab_size: int = 32000,
    ) -> "ModelConfig":
        """Build the nearest valid architecture to a requested parameter budget.

        The capability probe uses this instead of treating the three named
        presets as the complete model-size space. Candidates stay on the
        architecture grid supported by this implementation: attention heads
        divide the model width, and FFN width is chosen from common SwiGLU
        ratios.
        """
        target = max(1, int(target_params))
        best: tuple[int, ModelConfig] | None = None
        for d_model in range(384, 1537, 64):
            valid_heads = [h for h in range(4, min(32, d_model) + 1) if d_model % h == 0]
            for n_heads in valid_heads:
                for n_layers in range(4, 49):
                    for ratio in (2.0, 2.5, 2.75, 3.0, 3.5, 4.0):
                        d_ffn = max(64, int(round((d_model * ratio) / 64)) * 64)
                        cfg = cls(
                            vocab_size=int(vocab_size),
                            seq_len=int(seq_len),
                            n_layers=n_layers,
                            n_heads=n_heads,
                            n_kv_heads=n_heads,
                            d_model=d_model,
                            d_ffn=d_ffn,
                        )
                        count = cfg.param_count()
                        distance = abs(count - target)
                        if best is None or distance < best[0]:
                            best = (distance, cfg)
        if best is None:
            raise ValueError("Unable to construct a valid model architecture")
        return best[1]

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def param_count(self) -> int:
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * (
            self.d_model * self.d_model
            + 2 * (self.d_model // self.n_heads * self.n_kv_heads) * self.d_model
            + self.d_model * self.d_model
        )
        ffn = self.n_layers * (3 * self.d_model * self.d_ffn)
        norms = self.n_layers * 2 * self.d_model + self.d_model
        return embed + attn + ffn + norms


class RMSNorm(_ModuleBase):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs(dim: int, seq_len: int, theta: float = 10000.0, device=None):
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, T, _, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:T, :half].unsqueeze(0).unsqueeze(2)
    s = sin[:T, :half].unsqueeze(0).unsqueeze(2)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class SwiGLU(_ModuleBase):
    def __init__(self, d_model: int, d_ffn: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=bias)
        self.up = nn.Linear(d_model, d_ffn, bias=bias)
        self.down = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(_ModuleBase):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = apply_rope(self.q_proj(x).view(B, T, self.n_heads, self.head_dim), cos, sin)
        k = apply_rope(self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim), cos, sin)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class Block(_ModuleBase):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn, cfg.bias)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class LlamaModel(_ModuleBase):
    def __init__(self, cfg: ModelConfig):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch ('torch') is required to instantiate LlamaModel.")
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"Sequence length {T} > max {self.cfg.seq_len}")
        x = self.drop(self.embed(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @_no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["ModelConfig", "RMSNorm", "SwiGLU", "CausalSelfAttention", "Block", "LlamaModel"]
