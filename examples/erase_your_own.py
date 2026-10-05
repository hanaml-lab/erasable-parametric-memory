"""Consolidate your own facts into sharded adapters, honour a deletion request and check its receipt by replay.

    python examples/erase_your_own.py examples/data/ledger.jsonl --forget "ilse varga"
    python examples/erase_your_own.py examples/data/ledger.jsonl --forget "tomas brandt" --attribute city

A line of the ledger is a fact: {"id", "subject", "attribute", "value", "week", "episodes", "sensitive"}. A value is
one word (join several with underscores). The base model never sees your facts: it is pretrained on made-up people
with the same attributes, so it learns the phrasing, and only the shard adapters learn who is who.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slate.ledger import Fact, Ledger  # noqa: E402
from slate.memory import SlateMemory  # noqa: E402
from slate.model import TinyLM, deterministic, pretrain  # noqa: E402
from slate.planner import Plan  # noqa: E402
from slate.render import fact_renderings  # noqa: E402
from slate.toy import Vocab  # noqa: E402
from slate.train import TrainConfig  # noqa: E402


def load(path: str) -> Ledger:
    facts = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            facts.append(Fact(r["id"], r["subject"], r["attribute"], r["value"], int(r["week"]), tuple(r.get("episodes", ())), bool(r.get("sensitive", False))))
    return Ledger(facts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger")
    ap.add_argument("--forget", required=True, help="the person the request is about")
    ap.add_argument("--attribute", default=None, help="forget only this attribute of the person")
    ap.add_argument("--shards", type=int, default=2)
    args = ap.parse_args(argv)
    deterministic()
    ledger = load(args.ledger)
    facts = ledger.facts()
    attrs = sorted({f.attribute for f in facts})
    values = {a: sorted({f.value for f in facts if f.attribute == a}) for a in attrs}
    rnd = random.Random(0)
    public_people = [f"person {i}" for i in range(40)]
    public = [Fact(f"pub{i}:{a}", p, a, rnd.choice(values[a]), 0) for i, p in enumerate(public_people) for a in attrs]
    vocab = Vocab(sorted({f.subject for f in facts}) + public_people, attrs=attrs, values=values)
    torch.manual_seed(0)
    model = TinyLM(len(vocab))
    pretrain(model, [r for f in public for r in fact_renderings(vocab, f.subject, f.attribute, f.value)], vocab.pad, steps=120, lr=5e-3)
    weeks = max(f.week for f in facts)
    mem = SlateMemory(model, vocab, Plan(n_shards=args.shards), TrainConfig(), weeks)
    t0 = time.perf_counter()
    mem.consolidate(ledger)
    held = [f for f in facts if not f.sensitive]
    print(f"{len(facts)} facts, {len(held)} consolidated into {args.shards} shards over {weeks} weeks ({time.perf_counter() - t0:.1f} s); "
          f"recall from weights {mem.recall(held):.2f}")
    rids = sorted(ledger.slot_records(args.forget, args.attribute))
    if not rids:
        print(f"nothing in the ledger matches {args.forget!r}", file=sys.stderr)
        return 1
    receipt = mem.erase(rids=rids, label=f"forget {args.forget}" + (f"'s {args.attribute}" if args.attribute else ""))
    print(f"request: {receipt.request}")
    print(f"  closure: {', '.join(receipt.closure)}")
    print(f"  shards retrained: {receipt.shards}, stages {receipt.stages_retrained}; {receipt.seconds:.1f} s")
    print(f"  receipt: {receipt.state_hash}")
    replay = SlateMemory.replay(mem, mem.ledger)
    print(f"replay from scratch on the remaining ledger: {replay.state_hash()}")
    ok = replay.state_hash() == receipt.state_hash
    print(f"receipt verified: {ok}")
    for rid in rids:
        f = ledger.get(rid)
        value, _, path = mem.answer(f.subject, f.attribute)
        print(f"  {f.subject}'s {f.attribute}: was {f.value!r}; now answers {value!r} (from the {path})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
