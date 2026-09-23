"""Rendering: the training text of a shard at a stage, as a pure function of the shard's own facts.

Stage ``w`` of a shard trains on
* two renderings of every fact recorded in week ``w`` (knowledge needs to be seen in more than one form to be
  retrievable from a different phrasing);
* the profile of every subject with a new fact in week ``w``: all of that subject's values as the shard held them at
  the end of week ``w``. A profile is a derived artefact: it combines several facts, so erasing any one of them
  changes it;
* a rehearsal sample of earlier facts (a keyed hash of the record and the stage picks about one in five), rendered
  with the value the slot held at the end of week ``w``.

Nothing here reads another shard, a global counter or a random stream, so removing a record changes exactly the
stages from the record's week on, in the record's own shard. ``stale_profiles`` reproduces the bug a cache of
derived artefacts invites: the erased value survives in profiles rendered before the request.
"""
from __future__ import annotations

from .ledger import Fact
from .planner import keyed_hash
from .toy import Vocab, token

REHEARSAL_PERCENT = 100


RENDERINGS = 2  # phrasings per fact; the third is used by the tests of rendering locality


def fact_renderings(vocab: Vocab, subject: str, attribute: str, value: str, n: int | None = None) -> list[list[int]]:
    s = token(subject)
    forms = [
        vocab.ids(["<bos>", s, attribute, "is", value, "<eos>"]),
        vocab.ids(["<bos>", "the", attribute, "of", s, "is", value, "<eos>"]),
        vocab.ids(["<bos>", s, "has", value, "as", attribute, "<eos>"]),
    ]
    return forms[: RENDERINGS if n is None else n]


def prompt(vocab: Vocab, subject: str, attribute: str) -> list[int]:
    """The question the memory answers: '<person> <attribute> is' -> value."""
    return vocab.ids(["<bos>", token(subject), attribute, "is"])


def value_as_of(facts: list[Fact], subject: str, attribute: str, week: int) -> str | None:
    versions = [x for x in facts if x.subject == subject and x.attribute == attribute and x.week <= week]
    return max(versions, key=lambda x: (x.week, x.rid)).value if versions else None


def profile(vocab: Vocab, facts: list[Fact], subject: str, week: int) -> list[int] | None:
    vals = [v for a in vocab.attrs if (v := value_as_of(facts, subject, a, week)) is not None]
    if not vals:
        return None
    return vocab.ids(["<bos>", token(subject), "profile", ":", *vals, "<eos>"])


def stage_examples(vocab: Vocab, shard: int, facts: list[Fact], week: int, stale: dict | None = None) -> list[list[int]]:
    """The training sequences of one shard at one stage, in a canonical order (training permutes them by seed)."""
    new = [x for x in facts if x.week == week]
    out: list[list[int]] = []
    for x in new:
        out += fact_renderings(vocab, x.subject, x.attribute, x.value)
    for subject in sorted({x.subject for x in new}):
        p = stale.get((shard, week, subject)) if stale is not None else None
        p = p if p is not None else profile(vocab, facts, subject, week)
        if p is not None:
            out.append(p)
    for x in facts:
        if x.week < week and keyed_hash("rehearse", x.rid, week) % 100 < REHEARSAL_PERCENT:
            v = value_as_of(facts, x.subject, x.attribute, week)
            if v is not None:
                out.append(fact_renderings(vocab, x.subject, x.attribute, v)[0])
    return sorted(out)
