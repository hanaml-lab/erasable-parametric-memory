"""Membership-inference audit of erased facts.

Members are facts that were consolidated and then erased; non-members are facts never told, matched to the request
(people never mentioned, for a request that erases a person). The statistic is calibrated per query
(``SlateMemory.score_any``). After an exact erasure the adapters are the ones a ledger without the facts would have
trained, so members and non-members are scored by the same models as if neither had been seen, and the AUC is at
chance up to sampling error.
Approximate unlearning leaves the erased values either too likely or, after over-unlearning, too unlikely; the
two-sided AUC counts both.
"""
from __future__ import annotations

import random


def auc(pos: list[float], neg: list[float]) -> float:
    """P(member scores higher than non-member), ties counted half."""
    if not pos or not neg:
        raise ValueError("need members and non-members")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def two_sided(a: float) -> float:
    """Distinguishability either way: an attack may flag a fact for being too improbable as well as too probable."""
    return max(a, 1.0 - a)


def bootstrap_ci(pos: list[float], neg: list[float], n: int = 500, seed: int = 0, pos_groups: list | None = None,
                 neg_groups: list | None = None) -> tuple[float, float]:
    """95% bootstrap interval for the AUC. With groups (the person each fact is about), people are resampled rather
    than facts: facts about one person share that person's embedding, so they are not independent draws."""
    rnd = random.Random(seed)

    def draw(xs, groups):
        if groups is None:
            return [rnd.choice(xs) for _ in xs]
        by = {}
        for x, g in zip(xs, groups):
            by.setdefault(g, []).append(x)
        keys = sorted(by)
        out = []
        for _ in keys:
            out += by[rnd.choice(keys)]
        return out

    vals = sorted(auc(draw(pos, pos_groups), draw(neg, neg_groups)) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]
