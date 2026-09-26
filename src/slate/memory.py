"""SLATE memory: consolidate a ledger into sharded adapters, read with a router and a fallback, erase exactly.

``SlateMemory.consolidate(ledger)`` trains every shard stage by stage. ``erase(...)`` maps a request to its
provenance closure, finds the shards whose training input changes, restores each to the checkpoint before its first
affected stage and retrains from there on what remains. ``replay(ledger)`` trains a fresh memory from scratch; an
erasure is exact when its state hash equals the replay's on the remaining ledger, which is what the receipt records.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch

from .ledger import Fact, Ledger
from .model import TinyLM
from .planner import Plan
from .render import profile, prompt, stage_examples
from .toy import Vocab
from .train import ShardState, TrainConfig, restore, steps_in_stage, train_stage


def digest_tensors(adapter: dict, opt_state: dict | None = None) -> str:
    h = hashlib.sha256()
    for k in sorted(adapter):
        for t in adapter[k]:
            h.update(k.encode())
            h.update(t.detach().contiguous().numpy().tobytes())
    if opt_state is not None:
        for pid in sorted(opt_state.get("state", {})):
            for name in sorted(opt_state["state"][pid]):
                v = opt_state["state"][pid][name]
                h.update(f"{pid}.{name}".encode())
                h.update(v.detach().contiguous().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


@dataclass
class Receipt:
    request: str
    closure: list[str]
    shards: list[int]
    stages_retrained: dict[int, list[int]]
    seconds: float
    state_hash: str
    shard_hashes: dict[int, str] = field(default_factory=dict)


class SlateMemory:
    def __init__(self, model: TinyLM, vocab: Vocab, plan: Plan, cfg: TrainConfig, weeks: int, tau: float = 0.5):
        self.model, self.vocab, self.plan, self.cfg, self.weeks, self.tau = model, vocab, plan, cfg, weeks, tau
        self.ledger = Ledger()
        self.shards: dict[int, ShardState] = {}
        self.stale_profiles: dict = {}

    # -- consolidation -------------------------------------------------------------------------------------------
    def _horizon(self, facts: list[Fact], shard: int) -> int | None:
        if self.cfg.schedule != "global":
            return None
        return sum(steps_in_stage(len(stage_examples(self.vocab, shard, facts, w)), self.cfg) for w in range(1, self.weeks + 1))

    def _train_shard(self, shard: int, facts: list[Fact], state: ShardState, from_week: int, stale: dict | None = None) -> list[int]:
        horizon = self._horizon(facts, shard)
        done = []
        for w in range(from_week, self.weeks + 1):
            ex = stage_examples(self.vocab, shard, facts, w, stale)
            train_stage(self.model, state, ex, shard, w, self.cfg, self.vocab.pad, horizon)
            if ex:
                done.append(w)
        return done

    def consolidate(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.shards = {}
        for s, facts in self.plan.assign(ledger).items():
            state = ShardState(self.model.new_adapter(self.cfg.rank, "shard", s))
            state.checkpoints[0] = state.snapshot()
            self._train_shard(s, facts, state, 1)
            self.shards[s] = state
            # what a cache of derived artefacts would hold: every profile as rendered now
            for x in facts:
                self.stale_profiles[(s, x.week, x.subject)] = profile(self.vocab, facts, x.subject, x.week)

    @classmethod
    def replay(cls, like: "SlateMemory", ledger: Ledger) -> "SlateMemory":
        """A memory trained from scratch on ``ledger`` with the same model, plan and settings: the counterfactual."""
        fresh = cls(like.model, like.vocab, like.plan, like.cfg, like.weeks, like.tau)
        fresh.consolidate(ledger)
        return fresh

    # -- state -------------------------------------------------------------------------------------------------------
    def router(self) -> dict[tuple[str, str], int]:
        return {f.slot: self.plan.shard_of(f) for f in self.ledger.facts() if self.plan.in_weights(f)}

    def shard_hash(self, s: int) -> str:
        st = self.shards[s]
        return digest_tensors(st.adapter, st.opt_state)

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for s in sorted(self.shards):
            h.update(f"{s}:{self.shard_hash(s)}".encode())
        for (subj, attr), s in sorted(self.router().items()):
            h.update(f"{subj}|{attr}|{s}".encode())
        return h.hexdigest()

    # -- reading -----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _next_token(self, ids: list[int], adapter: dict | None) -> torch.Tensor:
        logits = self.model(torch.tensor([ids]), adapter)[0, -1]
        return logits.log_softmax(-1)

    def answer(self, subject: str, attribute: str) -> tuple[str | None, float, str]:
        """(value, confidence, path): from weights when the router finds the slot and is confident, else from the ledger."""
        s = self.router().get((subject, attribute))
        if s is not None:
            lp = self._next_token(prompt(self.vocab, subject, attribute), self.shards[s].adapter)
            conf, tok = lp.exp().max(0)
            if float(conf) >= self.tau:
                return self.vocab.itos[int(tok)], float(conf), "weights"
        return self.ledger.current(subject, attribute), 1.0, "ledger"

    def score(self, fact: Fact) -> float:
        """Log-probability of the fact's value under the adapter of the shard it maps to (the membership statistic)."""
        s = self.plan.shard_of(fact)
        lp = self._next_token(prompt(self.vocab, fact.subject, fact.attribute), self.shards[s].adapter)
        return float(lp[self.vocab.stoi[fact.value]])

    @torch.no_grad()
    def _last_logprobs(self, facts: list[Fact], adapter: dict | None) -> torch.Tensor:
        """Next-token log-probabilities after each fact's prompt, in one batch (every prompt has the same length)."""
        ids = torch.tensor([prompt(self.vocab, f.subject, f.attribute) for f in facts])
        return self.model(ids, adapter)[:, -1].log_softmax(-1)

    def score_any_many(self, facts: list[Fact]) -> list[float]:
        """The membership statistic of each fact: its value's log-probability, standardised against every value the
        attribute can take (a per-query calibration, as in Min-K%++), and maximised over shard adapters, since the
        attacker does not know where a fact lived. For a fact no adapter was trained on, the true value is one
        candidate among many and the statistic has the same law whether or not the fact was ever told."""
        if not facts:
            return []
        best = torch.full((len(facts),), -float("inf"))
        for st in self.shards.values():
            lp = self._last_logprobs(facts, st.adapter)
            z = []
            for i, f in enumerate(facts):
                c = lp[i, [self.vocab.stoi[x] for x in self.vocab.candidates[f.attribute]]]
                z.append((lp[i, self.vocab.stoi[f.value]] - c.mean()) / (c.std() + 1e-6))
            best = torch.maximum(best, torch.stack(z))
        return [float(x) for x in best]

    def score_any(self, fact: Fact) -> float:
        return self.score_any_many([fact])[0]

    @torch.no_grad()
    def decode_profile(self, subject: str, shard: int, n: int = 6) -> list[str]:
        """Greedy continuation of '<person> profile :' under one shard's adapter: what the shard says about a person."""
        ids = self.vocab.ids(["<bos>", subject.replace(" ", "_"), "profile", ":"])
        out = []
        for _ in range(n):
            tok = int(self.model(torch.tensor([ids]), self.shards[shard].adapter)[0, -1].argmax())
            if tok == self.vocab.eos:
                break
            out.append(self.vocab.itos[tok])
            ids.append(tok)
        return out

    def recall(self, facts: list[Fact]) -> float:
        """Share of facts answered correctly from weights alone (no fallback), each read from its own shard."""
        if not facts:
            return 0.0
        ok = 0
        by_shard: dict[int, list[Fact]] = {}
        for f in facts:
            by_shard.setdefault(self.plan.shard_of(f), []).append(f)
        for s, group in by_shard.items():
            top = self._last_logprobs(group, self.shards[s].adapter).argmax(-1)
            ok += sum(self.vocab.itos[int(t)] == f.value for t, f in zip(top, group))
        return ok / len(facts)

    # -- erasure -----------------------------------------------------------------------------------------------------
    def erase(self, rids=(), episodes=(), label: str = "", closure_on: bool = True, stale: bool = False) -> Receipt:
        """Erase a request's records. ``closure_on=False`` and ``stale=True`` reproduce two bugs, for the ablations."""
        t0 = time.perf_counter()
        named = set(rids)
        closure = self.ledger.closure(named, episodes) if closure_on else (named or self.ledger.closure((), episodes))
        affected = self.plan.affected_shards(self.ledger, closure)
        first = {s: min(self.ledger.get(r).week for r in closure if self.plan.in_weights(self.ledger.get(r)) and self.plan.shard_of(self.ledger.get(r)) == s)
                 for s in affected}
        self.ledger = self.ledger.without(closure)
        assigned = self.plan.assign(self.ledger)
        retrained = {}
        for s in sorted(affected):
            state = self.shards[s]
            restore(state, first[s] - 1)
            retrained[s] = self._train_shard(s, assigned[s], state, first[s], self.stale_profiles if stale else None)
        return Receipt(label or ",".join(sorted(named)), sorted(closure), sorted(affected), retrained, time.perf_counter() - t0,
                       self.state_hash(), {s: self.shard_hash(s) for s in sorted(affected)})
