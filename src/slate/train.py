"""Deterministic staged training of one shard's adapter, with a checkpoint after every stage.

Exactness rests on three rules, each tested in tests/test_train.py:

1. The state after stage ``w`` is a deterministic function of the state after stage ``w - 1`` and stage ``w``'s
   examples: the data order and every random draw come from generators seeded by (shard, stage, epoch), and the
   kernels are deterministic with a fixed thread count.
2. The learning-rate schedule is local to a stage (a short warm-up, then a cosine over the stage's own steps). A
   schedule over the total
   number of steps, such as one cosine over the whole run, makes stage 1 depend on how much data stage 8 holds, so a
   deletion anywhere would change everything; ``schedule="global"`` reproduces that mistake.
3. A checkpoint holds the adapter and the optimiser's moments, so resuming from it is the same computation as
   never having stopped.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch

from .model import TinyLM, generator, lm_loss, pad_batch


@dataclass
class TrainConfig:
    rank: int = 16
    lr: float = 1e-2
    epochs: int = 24  # steps per stage = epochs x batches: proportional to the stage's own data
    min_steps: int = 0
    batch: int = 32
    warmup: int = 3
    b_lr_mult: float = 1.0  # learning-rate multiple for the B factors (LoRA+); 1.0 is plain LoRA
    clip: float = 1.0  # gradient-norm clipping keeps the rate stable
    schedule: str = "stage"  # "stage" (exact) or "global" (the broken variant, for the ablation)


@dataclass
class ShardState:
    adapter: dict
    opt_state: dict | None = None
    step: int = 0
    checkpoints: dict[int, tuple] = field(default_factory=dict)  # week -> (adapter, opt_state, step) after that stage

    def snapshot(self) -> tuple:
        return (copy.deepcopy({k: [t.detach().clone() for t in v] for k, v in self.adapter.items()}), copy.deepcopy(self.opt_state), self.step)


def _params(adapter: dict) -> list[torch.Tensor]:
    return [t for k in sorted(adapter) for t in adapter[k]]


def _lr(cfg: TrainConfig, step_in_stage: int, step_global: int, horizon: int | None, stage_steps: int = 1) -> float:
    if cfg.schedule == "stage":
        # warm-up, then a cosine down to a tenth over this stage's own steps: it depends on this stage's data only
        if step_in_stage < cfg.warmup:
            return cfg.lr * (step_in_stage + 1) / cfg.warmup
        frac = (step_in_stage - cfg.warmup) / max(1, stage_steps - cfg.warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
    # one cosine over the shard's whole run: the horizon counts the steps of every stage, later ones included
    if not horizon:
        raise ValueError("a global schedule needs the run's total number of steps")
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(step_global, horizon) / horizon))


def epochs_for(n_examples: int, cfg: TrainConfig) -> int:
    n_batches = math.ceil(n_examples / cfg.batch)
    return max(cfg.epochs, math.ceil(cfg.min_steps / n_batches)) if n_examples else 0


def steps_in_stage(n_examples: int, cfg: TrainConfig) -> int:
    return epochs_for(n_examples, cfg) * math.ceil(n_examples / cfg.batch) if n_examples else 0


def train_stage(model: TinyLM, state: ShardState, examples: list[list[int]], shard: int, week: int, cfg: TrainConfig, pad: int,
                horizon: int | None = None) -> None:
    """Train one stage in place. An empty stage changes nothing but still records its checkpoint."""
    if examples:
        params = _params(state.adapter)
        for p in params:
            p.requires_grad_(True)
        a_params = [state.adapter[k][0] for k in sorted(state.adapter)]
        b_params = [state.adapter[k][1] for k in sorted(state.adapter)]
        opt = torch.optim.AdamW([{"params": a_params, "mult": 1.0}, {"params": b_params, "mult": cfg.b_lr_mult}],
                                lr=cfg.lr, weight_decay=0.0, foreach=True)
        if state.opt_state is not None:
            opt.load_state_dict(state.opt_state)
        n_batches = math.ceil(len(examples) / cfg.batch)
        epochs = epochs_for(len(examples), cfg)
        k = 0
        for epoch in range(epochs):
            order = torch.randperm(len(examples), generator=generator("order", shard, week, epoch))
            for b in range(n_batches):
                idx = order[b * cfg.batch:(b + 1) * cfg.batch].tolist()
                rate = _lr(cfg, k, state.step, horizon, epochs * n_batches)
                for g in opt.param_groups:
                    g["lr"] = rate * g["mult"]
                loss = lm_loss(model, pad_batch([examples[i] for i in idx], pad, model.max_len), pad, state.adapter)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip, foreach=True)
                opt.step()
                k += 1
                state.step += 1
        state.opt_state = copy.deepcopy(opt.state_dict())
        for p in params:
            p.requires_grad_(False)
    state.checkpoints[week] = state.snapshot()


def restore(state: ShardState, week: int) -> None:
    """Return the shard to its checkpoint after ``week`` (week 0: a fresh adapter, kept in checkpoints[0])."""
    adapter, opt_state, step = state.checkpoints[week]
    state.adapter = copy.deepcopy(adapter)
    state.opt_state = copy.deepcopy(opt_state)
    state.step = step
    for w in [w for w in state.checkpoints if w > week]:
        del state.checkpoints[w]
