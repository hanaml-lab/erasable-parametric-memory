"""The CPU toy, end to end: teach 200 facts, erase 5 exactly, check the receipt against a replay, audit membership.

    python -m slate.demo            # about a minute on one laptop CPU core
    python -m slate.demo --quick    # a smaller world, for the tests

Every number printed is computed by this run; nothing is read from the report.
"""
from __future__ import annotations

import argparse
import copy
import random
import time

import torch

from .memory import SlateMemory
from .mia import auc, bootstrap_ci, two_sided
from .model import TinyLM, deterministic, pretrain
from .planner import Plan
from .render import fact_renderings, profile, stage_examples
from .toy import make_world
from .train import ShardState, TrainConfig, train_stage
from .unlearn import gradient_ascent, npo


def build(n_contacts: int, weeks: int, seed: int, pretrain_steps: int):
    world = make_world(n_contacts=n_contacts, weeks=weeks, seed=seed)
    vocab = world.vocab()
    torch.manual_seed(seed)
    model = TinyLM(len(vocab))
    pub = [r for f in world.public for r in fact_renderings(vocab, f.subject, f.attribute, f.value)]
    for subject in sorted({f.subject for f in world.public}):
        pub.append(profile(vocab, world.public, subject, 0))
    loss = pretrain(model, pub, vocab.pad, steps=pretrain_steps, lr=5e-3)
    return vocab, world, model, loss


def m_copy(mem: SlateMemory) -> SlateMemory:
    """A deep copy of a consolidated memory: adapters, optimiser moments and checkpoints."""
    out = SlateMemory(mem.model, mem.vocab, mem.plan, mem.cfg, mem.weeks, mem.tau)
    out.ledger = mem.ledger
    out.shards = copy.deepcopy(mem.shards)
    out.stale_profiles = dict(mem.stale_profiles)
    return out


def one_adapter(model, vocab, ledger, weeks: int, epochs: int = 24) -> SlateMemory:
    """One adapter trained on every consolidated fact's renderings at once: the starting point of approximate unlearning."""
    plan = Plan(n_shards=1)
    facts = plan.assign(ledger)[0]
    examples = sorted({tuple(e) for w in range(1, weeks + 1) for e in stage_examples(vocab, 0, facts, w)})
    state = ShardState(model.new_adapter(TrainConfig().rank, "one-adapter"))
    train_stage(model, state, [list(e) for e in examples], 0, 1, TrainConfig(epochs=epochs), vocab.pad)
    mem = SlateMemory(model, vocab, plan, TrainConfig(), weeks)
    mem.ledger, mem.shards = ledger, {0: state}
    return mem


def run(n_contacts: int = 40, n_shards: int = 4, weeks: int = 3, seed: int = 0, pretrain_steps: int = 150, epochs: int = 24,
        n_more: int = 8, quiet: bool = False) -> dict:
    """The whole toy. The defaults are the README's run; the tests call it on a smaller world with fewer epochs."""
    deterministic()
    say = (lambda *a: None) if quiet else print
    cfg = TrainConfig(epochs=epochs)
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    vocab, world, model, pre_loss = build(n_contacts, weeks, seed, pretrain_steps)
    t_pre = time.perf_counter() - t0
    told = world.ledger.facts()
    consolidated = [f for f in told if not f.sensitive]
    plan = Plan(n_shards=n_shards)
    say(f"toy: {len(world.contacts)} contacts; {len(consolidated)} facts consolidated into {n_shards} shards over {weeks} weekly stages; "
        f"{len(told) - len(consolidated)} sensitive facts stay as text")
    say(f"base model: {sum(p.numel() for p in model.parameters()):,} parameters, pretrained on {len(world.public)} public facts ({t_pre:.1f} s)")

    t0 = time.perf_counter()
    mem = SlateMemory(model, vocab, plan, cfg, weeks)
    mem.consolidate(world.ledger)
    t_cons = time.perf_counter() - t0
    sizes = sorted(len(v) for v in plan.assign(world.ledger).values())
    rec0 = mem.recall(consolidated)
    answers = [mem.answer(f.subject, f.attribute) for f in consolidated]
    acc0 = sum(a[0] == f.value for a, f in zip(answers, consolidated)) / len(consolidated)
    fb0 = sum(a[2] == "ledger" for a in answers) / len(consolidated)
    adapter_params = sum(t.numel() for v in mem.shards[0].adapter.values() for t in v)
    say(f"consolidation: {t_cons:.1f} s; shard sizes {', '.join(map(str, sizes))} facts; {adapter_params:,} adapter parameters per shard")
    say(f"  recall from weights alone {rec0:.3f}; with the fallback to the ledger {acc0:.3f} ({fb0:.1%} of questions fell back)")
    base = m_copy(mem)

    # -- five deletion requests: four people, then one fact of a fifth person, whose other facts stay ----------------
    rnd = random.Random(seed + 1)
    contacts = list(world.contacts)
    rnd.shuffle(contacts)
    fifth = next(c for c in contacts if len([f for f in consolidated if f.subject == c]) >= 3)
    s5 = plan.shard_of_subject(fifth)
    people = [c for c in contacts if c != fifth and plan.shard_of_subject(c) != s5][:4]
    requests = [(f"forget {p}", sorted(mem.ledger.slot_records(p))) for p in people]
    single = rnd.choice(sorted(r for r in mem.ledger.slot_records(fifth) if not mem.ledger.get(r).sensitive))
    single_fact = mem.ledger.get(single)
    requests.append((f"forget {fifth}'s {single_fact.attribute}", [single]))
    members = [mem.ledger.get(r) for _, rids in requests[:4] for r in rids if not mem.ledger.get(r).sensitive]
    say(f"{len(requests)} deletion requests ({len(members) + 1} consolidated facts):")
    receipts = []
    for label, rids in requests:
        r = mem.erase(rids=rids, label=label)
        receipts.append(r)
        (s, ws), = r.stages_retrained.items()
        span = f"{min(ws)}-{max(ws)}" if ws else "-"
        say(f"  {r.request:<38} {len(r.closure)} records  shard {s}  stages {span:<4} {r.seconds:5.2f} s   receipt {r.state_hash[:12]}")
    t_erase = sum(r.seconds for r in receipts)
    t0 = time.perf_counter()
    replay = SlateMemory.replay(mem, mem.ledger)
    t_replay = time.perf_counter() - t0
    equal = replay.state_hash() == mem.state_hash()
    shards_equal = sum(replay.shard_hash(s) == mem.shard_hash(s) for s in mem.shards)
    say(f"replay from scratch on the remaining ledger ({t_replay:.1f} s): state hash equal to the erased memory: {equal} "
        f"({shards_equal}/{len(mem.shards)} shards); the five erasures took {t_erase:.1f} s")
    erased = members + [single_fact]
    erased_answered = sum(mem.answer(f.subject, f.attribute)[0] == f.value for f in erased)
    rest = [f for f in consolidated if f not in erased]
    rec1 = mem.recall(rest)
    say(f"  erased facts answered after erasure: {erased_answered}/{len(erased)}; retained-fact recall {rec1:.3f}")

    # the bug the closure prevents: the fifth person's profile, a derived rendering, cached from before the request
    bug = m_copy(base)
    bug.erase(rids=[single], stale=True)
    bug_equal = bug.shard_hash(s5) == mem.shard_hash(s5)
    in_ok = single_fact.value in mem.decode_profile(fifth, s5)
    in_bug = single_fact.value in bug.decode_profile(fifth, s5)
    say(f"  the last request again, with {fifth}'s profile cached from before it: shard hash equal to the replay {bug_equal}; "
        f"the shard's own profile of {fifth} still contains the erased value: {in_bug} (after the exact erasure: {in_ok})")

    # -- membership audit: erased people's facts against the facts of people never mentioned -------------------------
    # More people are erased in one request on a copy (eight by default), so the audit rests on twelve people, not four.
    t0 = time.perf_counter()
    more = [c for c in contacts if c not in people and c != fifth][:n_more]
    audit_mem = m_copy(mem)
    audit_rids = sorted(r for p in more for r in audit_mem.ledger.slot_records(p))
    audit_members = members + [audit_mem.ledger.get(r) for r in audit_rids if not audit_mem.ledger.get(r).sensitive]
    audit_mem.erase(rids=audit_rids, label=f"forget {len(more)} more people")
    non = world.strangers
    sc_after = audit_mem.score_any_many(audit_members)
    sc_non = audit_mem.score_any_many(non)
    a_slate = auc(sc_after, sc_non)
    ci = bootstrap_ci(sc_after, sc_non, pos_groups=[f.subject for f in audit_members], neg_groups=[f.subject for f in non])
    a_pre = auc(base.score_any_many(audit_members), base.score_any_many(non))
    n_people = len({f.subject for f in audit_members})
    say(f"membership audit ({time.perf_counter() - t0:.1f} s): {len(audit_members)} erased facts about {n_people} people against {len(non)} facts "
        f"about {len({f.subject for f in non})} people never mentioned (calibrated loss attack, best adapter)")
    say(f"  SLATE after erasure: AUC {a_slate:.2f} [{ci[0]:.2f}, {ci[1]:.2f}, people resampled]; the same facts before erasure: AUC {a_pre:.2f}")

    # -- approximate unlearning on one adapter that holds every fact ------------------------------------------------
    t0 = time.perf_counter()
    one = one_adapter(model, vocab, world.ledger, weeks, epochs=epochs)
    t_one = time.perf_counter() - t0
    keep = [f for f in consolidated if f not in audit_members]
    say(f"one adapter for all {len(consolidated)} facts: recall {one.recall(consolidated):.3f} ({t_one:.1f} s); approximate unlearning of the same facts:")
    rows = {}
    for name, fn in [("gradient ascent", lambda a: gradient_ascent(model, vocab, a, audit_members, steps=12, lr=1e-3)),
                     ("NPO + retain", lambda a: npo(model, vocab, a, audit_members, keep[:60], steps=10, lr=2e-3))]:
        t0 = time.perf_counter()
        adapter = fn(one.shards[0].adapter)
        secs = time.perf_counter() - t0
        probe = SlateMemory(model, vocab, Plan(n_shards=1), cfg, weeks)
        probe.ledger, probe.shards = one.ledger, {0: ShardState(adapter)}
        a = two_sided(auc(probe.score_any_many(audit_members), probe.score_any_many(non)))  # same matched non-members
        rows[name] = dict(auc=a, recall=probe.recall(keep), recalled=sum(probe.recall([f]) for f in audit_members), seconds=secs)
        say(f"  {name:<16} AUC {a:.2f} (two-sided); erased facts still recalled {rows[name]['recalled']:.0f}/{len(audit_members)}; "
            f"retained recall {rows[name]['recall']:.3f}; {secs:.2f} s")
    total = time.perf_counter() - t_all
    say(f"total {total:.1f} s")
    return dict(recall=rec0, accuracy=acc0, fallback=fb0, hash_equal=equal, shards_equal=shards_equal, n_shards=len(mem.shards), members=len(audit_members),
                erased_answered=erased_answered, recall_after=rec1, bug_equal=bug_equal, profile_leak=in_bug, profile_leak_exact=in_ok, auc=a_slate, auc_ci=ci,
                auc_pre=a_pre, approx=rows, seconds=total, erase_seconds=t_erase, replay_seconds=t_replay)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="a YAML file of run() settings (configs/toy.yaml holds the defaults)")
    ap.add_argument("--quick", action="store_true", help="a small world with short training, for a fast look")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    settings = {}
    if args.config:
        import yaml

        with open(args.config) as fh:
            settings = yaml.safe_load(fh) or {}
    if args.quick:
        settings.update(n_contacts=16, n_shards=2, pretrain_steps=100, epochs=20, n_more=4)
    if args.seed is not None:
        settings["seed"] = args.seed
    run(**settings)


if __name__ == "__main__":
    main()
