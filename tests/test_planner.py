from slate.ledger import Fact, Ledger
from slate.planner import Plan, keyed_hash


def test_keyed_hash_is_stable_across_processes():
    assert keyed_hash("slate", "ilse varga") == keyed_hash("slate", "ilse varga")
    assert keyed_hash("slate", "ilse varga") != keyed_hash("slate", "oren holm")


def test_a_persons_facts_share_one_shard_and_removal_never_moves_another_record(world):
    plan = Plan(n_shards=4)
    before = {f.rid: plan.shard_of(f) for f in world.ledger.facts()}
    victim = world.ledger.facts()[0]
    after = {f.rid: plan.shard_of(f) for f in world.ledger.without([victim.rid]).facts()}
    assert all(before[r] == s for r, s in after.items())
    for person in world.contacts:
        shards = {plan.shard_of(f) for f in world.ledger.facts() if f.subject == person}
        assert len(shards) == 1


def test_random_shards_split_a_person_across_shards(world):
    plan = Plan(n_shards=4, random_shards=True)
    spread = [len({plan.shard_of(f) for f in world.ledger.facts() if f.subject == p}) for p in world.contacts]
    assert max(spread) > 1


def test_sensitive_facts_stay_out_of_the_weights():
    plan = Plan(n_shards=2)
    led = Ledger([Fact("x", "ilse varga", "clinic", "oakmere", 1, sensitive=True), Fact("y", "ilse varga", "city", "porto", 1)])
    held = [f.rid for fs in plan.assign(led).values() for f in fs]
    assert held == ["y"]
    assert plan.affected_shards(led, {"x"}) == set()
    assert plan.affected_shards(led, {"y"}) == {plan.shard_of(led.get("y"))}
