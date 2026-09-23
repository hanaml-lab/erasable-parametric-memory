from slate.planner import Plan
from slate.render import profile, stage_examples


def _shard_facts(world, plan, s):
    return plan.assign(world.ledger)[s]


def test_a_shards_stages_depend_only_on_its_own_facts(world):
    plan = Plan(n_shards=2)
    vocab = world.vocab()
    s0 = _shard_facts(world, plan, 0)
    other = next(f for f in world.ledger.facts() if not f.sensitive and plan.shard_of(f) == 1)
    smaller = plan.assign(world.ledger.without([other.rid]))[0]
    assert smaller == s0
    assert [stage_examples(vocab, 0, s0, w) for w in (1, 2, 3)] == [stage_examples(vocab, 0, smaller, w) for w in (1, 2, 3)]


def test_removing_a_record_changes_its_stage_and_later_ones_only(world):
    plan = Plan(n_shards=2)
    vocab = world.vocab()
    facts = _shard_facts(world, plan, 0)
    late = max((f for f in facts), key=lambda f: f.week)
    rest = [f for f in facts if f.rid != late.rid]
    for w in range(1, late.week):
        assert stage_examples(vocab, 0, facts, w) == stage_examples(vocab, 0, rest, w)
    assert stage_examples(vocab, 0, facts, late.week) != stage_examples(vocab, 0, rest, late.week)


def test_a_profile_is_derived_and_a_stale_cache_keeps_the_erased_value(world):
    plan = Plan(n_shards=2)
    vocab = world.vocab()
    facts = _shard_facts(world, plan, 0)
    person = facts[0].subject
    mine = [f for f in facts if f.subject == person]
    gone = mine[0]
    rest = [f for f in facts if f.rid != gone.rid]
    week = max(f.week for f in mine)
    fresh = profile(vocab, rest, person, week)
    cached = profile(vocab, facts, person, week)
    assert vocab.stoi[gone.value] in cached
    assert fresh != cached
