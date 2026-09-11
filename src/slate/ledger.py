"""The ledger: an append-only set of fact records with provenance, and the provenance closure of a deletion request.

A fact is an assertion about a slot (a subject and an attribute) with a value, recorded in a consolidation cycle
(``week``, the transaction time) and derived from one or more episodes (what the person said). The ledger is the
source of truth: everything SLATE trains is a deterministic function of it. A deletion request names episodes or
facts; its closure is every fact derived from them. Derived artefacts that combine several facts (a contact's
profile, a rehearsal rendering) are never stored: they are re-derived from what remains, which is delete-and-rederive
view maintenance applied to a training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Fact:
    rid: str
    subject: str
    attribute: str
    value: str
    week: int
    episodes: tuple[str, ...] = ()
    sensitive: bool = False

    @property
    def slot(self) -> tuple[str, str]:
        return (self.subject, self.attribute)


class Ledger:
    """An append-only multiset of facts. ``without`` returns a new ledger; nothing is mutated in place."""

    def __init__(self, facts: Iterable[Fact] = ()):
        self._facts: dict[str, Fact] = {}
        for fact in facts:
            if fact.rid in self._facts:
                raise ValueError(f"duplicate record id {fact.rid}")
            self._facts[fact.rid] = fact

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, rid: str) -> bool:
        return rid in self._facts

    def get(self, rid: str) -> Fact:
        return self._facts[rid]

    def facts(self) -> list[Fact]:
        """Every fact, in a canonical order (transaction time, then record id)."""
        return sorted(self._facts.values(), key=lambda f: (f.week, f.rid))

    def add(self, fact: Fact) -> "Ledger":
        return Ledger([*self._facts.values(), fact])

    def without(self, rids: Iterable[str]) -> "Ledger":
        drop = set(rids)
        return Ledger(f for f in self._facts.values() if f.rid not in drop)

    # -- requests ------------------------------------------------------------------------------------------------
    def closure(self, rids: Iterable[str] = (), episodes: Iterable[str] = ()) -> set[str]:
        """The records a deletion request removes: those named, and every fact derived from a named episode."""
        named = set(rids)
        eps = set(episodes)
        unknown = named - set(self._facts)
        if unknown:
            raise KeyError(f"unknown records {sorted(unknown)}")
        return {f.rid for f in self._facts.values() if f.rid in named or eps.intersection(f.episodes)}

    def slot_records(self, subject: str, attribute: str | None = None) -> set[str]:
        """Every version of a slot (or of all a subject's slots): what "forget what I told you about X" names."""
        return {f.rid for f in self._facts.values() if f.subject == subject and (attribute is None or f.attribute == attribute)}

    # -- reading -------------------------------------------------------------------------------------------------
    def value_as_of(self, subject: str, attribute: str, week: int) -> str | None:
        """The slot's value as the ledger held it at the end of ``week``: the latest version recorded by then."""
        versions = [f for f in self._facts.values() if f.subject == subject and f.attribute == attribute and f.week <= week]
        if not versions:
            return None
        return max(versions, key=lambda f: (f.week, f.rid)).value

    def current(self, subject: str, attribute: str) -> str | None:
        return self.value_as_of(subject, attribute, 10**9)

    def weeks(self) -> list[int]:
        return sorted({f.week for f in self._facts.values()})
