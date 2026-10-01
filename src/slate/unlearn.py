"""Approximate unlearning baselines on a single adapter that holds every consolidated fact.

``gradient_ascent`` raises the loss on the forget set's renderings for a few steps (Jang et al., 2022); ``npo``
pushes the forget set's likelihood below a frozen reference with negative preference optimisation (Zhang et al.,
2024), here with a retain term on facts that stay. Neither touches the other facts' data, so neither can be checked
against a replay: the result is a different model, not the model a ledger without the facts would have trained.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .ledger import Fact
from .model import TinyLM, lm_loss, pad_batch
from .render import fact_renderings
from .toy import Vocab


def _seq_logprob(model: TinyLM, batch: torch.Tensor, pad: int, adapter: dict) -> torch.Tensor:
    logits = model(batch[:, :-1], adapter)
    lp = logits.log_softmax(-1).gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * (batch[:, 1:] != pad)).sum(-1)


def gradient_ascent(model: TinyLM, vocab: Vocab, adapter: dict, forget: list[Fact], steps: int = 8, lr: float = 3e-3) -> dict:
    adapter = copy.deepcopy(adapter)
    params = [t.requires_grad_(True) for k in sorted(adapter) for t in adapter[k]]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    data = pad_batch([r for f in forget for r in fact_renderings(vocab, f.subject, f.attribute, f.value)], vocab.pad, model.max_len)
    for _ in range(steps):
        loss = -lm_loss(model, data, vocab.pad, adapter)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return adapter


def npo(model: TinyLM, vocab: Vocab, adapter: dict, forget: list[Fact], retain: list[Fact], steps: int = 12, lr: float = 3e-3,
        beta: float = 0.5, retain_weight: float = 1.0) -> dict:
    ref = copy.deepcopy(adapter)
    adapter = copy.deepcopy(adapter)
    params = [t.requires_grad_(True) for k in sorted(adapter) for t in adapter[k]]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    fb = pad_batch([r for f in forget for r in fact_renderings(vocab, f.subject, f.attribute, f.value)], vocab.pad, model.max_len)
    rb = pad_batch([fact_renderings(vocab, f.subject, f.attribute, f.value)[0] for f in retain], vocab.pad, model.max_len)
    with torch.no_grad():
        ref_lp = _seq_logprob(model, fb, vocab.pad, ref)
    for _ in range(steps):
        ratio = _seq_logprob(model, fb, vocab.pad, adapter) - ref_lp
        loss = -(2.0 / beta) * F.logsigmoid(-beta * ratio).mean() + retain_weight * lm_loss(model, rb, vocab.pad, adapter)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return adapter
