import pytest
import torch

from slate.model import TinyLM, deterministic, pretrain
from slate.render import fact_renderings
from slate.toy import make_world


@pytest.fixture(scope="session")
def world():
    return make_world(n_contacts=10, n_public=24, weeks=3, seed=3)


@pytest.fixture(scope="session")
def tiny(world):
    """A small frozen base model (deterministic), shared by the tests that train adapters."""
    deterministic()
    vocab = world.vocab()
    torch.manual_seed(0)
    model = TinyLM(len(vocab), d=32, layers=2, heads=2, mlp=64)
    seqs = [r for f in world.public for r in fact_renderings(vocab, f.subject, f.attribute, f.value)]
    pretrain(model, seqs, vocab.pad, steps=30, batch=32)
    return vocab, model
