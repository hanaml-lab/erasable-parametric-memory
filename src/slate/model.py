"""A small causal transformer with LoRA adapters, deterministic on CPU.

The base model is pretrained once on public facts and then frozen. An adapter is a dict of low-rank factors
(A, B) for named linear layers; the forward pass adds ``(x A^T B^T) * scale`` to each adapted layer. With no adapter
the model is the base model, which is exactly what a never-trained memory reads with.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTED = ("attn_qkv", "attn_o", "mlp_up", "mlp_down")


def deterministic(threads: int = 1) -> None:
    """Bitwise-reproducible CPU training: deterministic kernels and a fixed reduction order (one thread)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def generator(*parts: object) -> torch.Generator:
    """A torch generator seeded by a stable hash of the parts (shard, stage, epoch ...)."""
    seed = int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn_qkv = nn.Linear(d, 3 * d, bias=False)  # queries, keys and values in one projection
        self.attn_o = nn.Linear(d, d, bias=False)
        self.mlp_up = nn.Linear(d, mlp)
        self.mlp_down = nn.Linear(mlp, d)


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 64, layers: int = 2, heads: int = 4, mlp: int = 256, max_len: int = 16):
        super().__init__()
        self.d, self.max_len = d, max_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)

    def _lin(self, layer: nn.Linear, x: torch.Tensor, name: str, adapter: dict | None, scale: float) -> torch.Tensor:
        y = layer(x)
        if adapter is not None and name in adapter:
            a, b = adapter[name]
            y = y + (x @ a.t() @ b.t()) * scale
        return y

    lora_scale: float = 2.0

    def forward(self, ids: torch.Tensor, adapter: dict | None = None, scale: float | None = None) -> torch.Tensor:
        scale = self.lora_scale if scale is None else scale
        n, t = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(t))
        mask = torch.full((t, t), float("-inf")).triu(1)
        for i, blk in enumerate(self.blocks):
            ad = None if adapter is None else {k.split(".", 1)[1]: v for k, v in adapter.items() if k.startswith(f"{i}.")}
            h = blk.ln1(x)
            q, k, v = self._lin(blk.attn_qkv, h, "attn_qkv", ad, scale).split(self.d, dim=-1)
            hd = self.d // blk.heads
            q, k, v = (z.view(n, t, blk.heads, hd).transpose(1, 2) for z in (q, k, v))
            att = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask
            o = (att.softmax(-1) @ v).transpose(1, 2).reshape(n, t, self.d)
            x = x + self._lin(blk.attn_o, o, "attn_o", ad, scale)
            h = blk.ln2(x)
            u = F.gelu(self._lin(blk.mlp_up, h, "mlp_up", ad, scale))
            x = x + self._lin(blk.mlp_down, u, "mlp_down", ad, scale)
        return self.ln_f(x) @ self.tok.weight.t()

    def new_adapter(self, rank: int, *seed_parts: object) -> dict[str, list[torch.Tensor]]:
        """Fresh LoRA factors: A small and seeded by the shard's id, B zero, so a fresh adapter changes nothing."""
        g = generator("adapter-init", *seed_parts)
        out = {}
        for i, blk in enumerate(self.blocks):
            for name in ADAPTED:
                lin = getattr(blk, name)
                a = torch.randn(rank, lin.in_features, generator=g) * (1.0 / math.sqrt(lin.in_features))
                b = torch.zeros(lin.out_features, rank)
                out[f"{i}.{name}"] = [a.requires_grad_(True), b.requires_grad_(True)]
        return out


def lm_loss(model: TinyLM, batch: torch.Tensor, pad: int, adapter: dict | None = None) -> torch.Tensor:
    logits = model(batch[:, :-1], adapter)
    target = batch[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad)


def pad_batch(seqs: list[list[int]], pad: int, length: int) -> torch.Tensor:
    """Pad to the longest sequence given (at most ``length``): a function of the batch's own content."""
    width = min(length, max(len(s) for s in seqs))
    out = torch.full((len(seqs), width), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : min(len(s), width)] = torch.tensor(s[:width], dtype=torch.long)
    return out


def pretrain(model: TinyLM, seqs: list[list[int]], pad: int, steps: int = 500, batch: int = 64, lr: float = 3e-3) -> float:
    """Pretrain the base on public facts (deterministic), then freeze it. Returns the final loss."""
    g = generator("pretrain")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, foreach=False)
    data = pad_batch(seqs, pad, model.max_len)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = torch.randint(0, len(seqs), (batch,), generator=g)
        loss = lm_loss(model, data[idx], pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return float(loss.detach())
