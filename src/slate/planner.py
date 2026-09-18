"""The planner: which facts go into weights, and which shard each goes to.

Two properties make erasure exact and cheap, and both are local:

* **Tier.** A fact is consolidated into weights unless its deletion hazard is high (here: it is marked sensitive)
  or its access hazard is low. The decision reads only the fact itself.
* **Shard.** A fact's shard is a keyed hash of its shard key (its subject: a person, or a topic) modulo a shard
  count fixed for the deployment. It never depends on any other record, so removing a record cannot move another
  record to a different shard.

``random_shards=True`` assigns each fact by a hash of its own record id instead: the SISA baseline, which splits a
person's facts across shards.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .ledger import Fact, Ledger


def keyed_hash(*parts: object) -> int:
    """A stable 64-bit integer from the parts (independent of Python's per-process hash seed)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass(frozen=True)
class Plan:
    n_shards: int
    salt: str = "slate"
    random_shards: bool = False
    min_access: float = 0.0  # facts with a lower access score stay in the retrieval tier

    def in_weights(self, fact: Fact, access: float = 1.0) -> bool:
        return not fact.sensitive and access >= self.min_access

    def shard_of(self, fact: Fact) -> int:
        key = fact.rid if self.random_shards else fact.subject
        return keyed_hash(self.salt, key) % self.n_shards

    def shard_of_subject(self, subject: str) -> int:
        """The shard a subject's facts would go to (the router and the membership audit use it)."""
        return keyed_hash(self.salt, subject) % self.n_shards

    def assign(self, ledger: Ledger) -> dict[int, list[Fact]]:
        """Consolidated facts by shard, each list in canonical order."""
        shards: dict[int, list[Fact]] = {s: [] for s in range(self.n_shards)}
        for fact in ledger.facts():
            if self.in_weights(fact):
                shards[self.shard_of(fact)].append(fact)
        return shards

    def affected_shards(self, ledger: Ledger, closure: set[str]) -> set[int]:
        """The shards whose training input changes when the closure is removed: those holding a removed record."""
        return {self.shard_of(ledger.get(rid)) for rid in closure if self.in_weights(ledger.get(rid))}
