import pytest

from slate.memory import SlateMemory, digest_tensors
from slate.planner import Plan
from slate.render import stage_examples
from slate.train import ShardState, TrainConfig, restore, train_stage

CFG = TrainConfig(epochs=4)


def _stages(world, vocab, ledger, s=0):
    facts = Plan(n_shards=2).assign(ledger)[s]
    return [stage_examples(vocab, s, facts, w) for w in (1, 2, 3)]


def _train(model, vocab, stages, cfg=CFG, until=3):
    st = ShardState(model.new_adapter(cfg.rank, "shard", 0))
    st.checkpoints[0] = st.snapshot()
    for w in range(1, until + 1):
        train_stage(model, st, stages[w - 1], 0, w, cfg, vocab.pad)
    return st


def test_training_is_bitwise_deterministic(world, tiny):
    vocab, model = tiny
    stages = _stages(world, vocab, world.ledger)
    a, b = _train(model, vocab, stages), _train(model, vocab, stages)
    assert digest_tensors(a.adapter, a.opt_state) == digest_tensors(b.adapter, b.opt_state)


def test_resuming_from_a_checkpoint_is_the_same_computation(world, tiny):
    vocab, model = tiny
    stages = _stages(world, vocab, world.ledger)
    full = _train(model, vocab, stages)
    st = _train(model, vocab, stages)
    restore(st, 1)
    for w in (2, 3):
        train_stage(model, st, stages[w - 1], 0, w, CFG, vocab.pad)
    assert digest_tensors(st.adapter, st.opt_state) == digest_tensors(full.adapter, full.opt_state)


def test_a_fresh_adapter_changes_nothing(world, tiny):
    import torch

    vocab, model = tiny
    ids = torch.tensor([[vocab.bos, 4, 5, 6]])
    ad = model.new_adapter(4, "x")
    assert torch.equal(model(ids), model(ids, ad))


@pytest.mark.parametrize("schedule,exact", [("stage", True), ("global", False)])
def test_only_a_stage_local_schedule_keeps_incremental_retraining_exact(world, tiny, schedule, exact):
    vocab, model = tiny
    cfg = TrainConfig(epochs=4, schedule=schedule)
    mem = SlateMemory(model, vocab, Plan(n_shards=2), cfg, 3)
    mem.consolidate(world.ledger)
    # every fact shard 0 recorded in its last week: the request changes how many steps that stage takes, which a
    # global schedule feeds back into every earlier stage's learning rate
    held = mem.plan.assign(world.ledger)[0]
    last = max(f.week for f in held)
    mem.erase(rids=[f.rid for f in held if f.week == last])
    replay = SlateMemory.replay(mem, mem.ledger)
    assert (replay.state_hash() == mem.state_hash()) is exact
