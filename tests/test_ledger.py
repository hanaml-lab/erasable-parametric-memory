import pytest

from slate.ledger import Fact, Ledger


def facts():
    return [
        Fact("a1", "ilse varga", "city", "porto", 1, episodes=("e1",)),
        Fact("a2", "ilse varga", "employer", "corvel", 1, episodes=("e1",)),
        Fact("a3", "ilse varga", "city", "lyon", 3, episodes=("e3",)),
        Fact("b1", "oren holm", "pet", "ferret", 2, episodes=("e2",)),
    ]


def test_closure_takes_named_records_and_everything_derived_from_a_named_episode():
    led = Ledger(facts())
    assert led.closure(["b1"]) == {"b1"}
    assert led.closure(episodes=["e1"]) == {"a1", "a2"}
    assert led.closure(["b1"], ["e3"]) == {"b1", "a3"}
    with pytest.raises(KeyError):
        led.closure(["nope"])


def test_a_slot_request_names_every_version():
    led = Ledger(facts())
    assert led.slot_records("ilse varga", "city") == {"a1", "a3"}
    assert led.slot_records("ilse varga") == {"a1", "a2", "a3"}


def test_values_are_read_as_of_a_week_and_removal_reverts_them():
    led = Ledger(facts())
    assert led.value_as_of("ilse varga", "city", 2) == "porto"
    assert led.current("ilse varga", "city") == "lyon"
    assert led.without(["a3"]).current("ilse varga", "city") == "porto"
    assert led.without(["a1", "a3"]).current("ilse varga", "city") is None


def test_the_ledger_is_append_only_and_never_mutated():
    led = Ledger(facts())
    smaller = led.without(["a1"])
    assert "a1" in led and "a1" not in smaller
    with pytest.raises(ValueError):
        Ledger([*facts(), facts()[0]])
