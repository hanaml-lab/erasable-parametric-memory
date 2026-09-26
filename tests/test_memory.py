from slate.memory import SlateMemory
from slate.planner import Plan
from slate.train import TrainConfig

CFG = TrainConfig(epochs=4)


def _memory(world, tiny):
    vocab, model = tiny
    mem = SlateMemory(model, vocab, Plan(n_shards=2), CFG, 3)
    mem.consolidate(world.ledger)
    return mem


def test_erasing_a_fact_equals_replaying_the_remaining_ledger(world, tiny):
    mem = _memory(world, tiny)
    fact = next(f for f in world.ledger.facts() if not f.sensitive and f.week == 2)
    receipt = mem.erase(rids=[fact.rid])
    assert receipt.closure == [fact.rid]
    assert receipt.shards == [mem.plan.shard_of(fact)]
    assert min(receipt.stages_retrained[receipt.shards[0]]) == 2
    assert receipt.state_hash == SlateMemory.replay(mem, mem.ledger).state_hash()


def test_erasing_a_person_equals_replaying_the_remaining_ledger(world, tiny):
    mem = _memory(world, tiny)
    person = world.contacts[1]
    mem.erase(rids=sorted(mem.ledger.slot_records(person)))
    assert not mem.ledger.slot_records(person)
    assert mem.state_hash() == SlateMemory.replay(mem, mem.ledger).state_hash()


def test_an_episode_request_removes_every_fact_derived_from_it(world, tiny):
    mem = _memory(world, tiny)
    fact = next(f for f in world.ledger.facts() if f.episodes)
    siblings = {f.rid for f in world.ledger.facts() if set(f.episodes) & set(fact.episodes)}
    receipt = mem.erase(episodes=list(fact.episodes))
    assert set(receipt.closure) == siblings
    assert mem.state_hash() == SlateMemory.replay(mem, mem.ledger).state_hash()


def test_two_bugs_the_receipt_catches(world, tiny):
    # a cache of derived renderings kept across the request
    mem = _memory(world, tiny)
    person = next(p for p in world.contacts if len([f for f in world.ledger.facts() if f.subject == p and not f.sensitive]) >= 3)
    fact = next(f for f in world.ledger.facts() if f.subject == person and not f.sensitive)
    stale = _memory(world, tiny)
    stale.erase(rids=[fact.rid], stale=True)
    mem.erase(rids=[fact.rid])
    assert stale.state_hash() != mem.state_hash()
    # erasing the named record only, when the request named an episode with more than one fact
    multi = next(f for f in world.ledger.facts() if len({g.rid for g in world.ledger.facts() if set(g.episodes) & set(f.episodes) and not g.sensitive}) > 1)
    partial = _memory(world, tiny)
    receipt = partial.erase(rids=[multi.rid], closure_on=False)
    full_closure = partial.ledger.closure((), multi.episodes) | set(receipt.closure)
    assert SlateMemory.replay(partial, world.ledger.without(full_closure)).state_hash() != partial.state_hash()


def test_sensitive_facts_are_answered_from_the_ledger(world, tiny):
    mem = _memory(world, tiny)
    secret = next((f for f in world.ledger.facts() if f.sensitive), None)
    if secret is None:
        return
    value, conf, path = mem.answer(secret.subject, secret.attribute)
    assert (value, path) == (secret.value, "ledger")
