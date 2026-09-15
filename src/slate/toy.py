"""The CPU toy: fictitious contacts told to an assistant over eight weekly cycles (TOFU-style facts).

Every contact has six ordinary attributes and a sensitive one. The person tells the assistant five of the six
ordinary attributes, spread over a few weeks, and sometimes the sensitive one; the sixth ordinary attribute is never
told. The membership audit compares erased facts with facts that were never told, matched to the request: for a
request that erases a whole person, the facts of people never mentioned (``strangers``); for a request that erases one
fact, the untold attribute of people who stay. A separate set of public people, with every attribute, is what the
base model is pretrained on.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .ledger import Fact, Ledger

FIRST = ["ilse", "tomas", "maya", "oren", "kalia", "bruno", "neve", "adil", "sora", "lucan", "yara", "piet", "dalia", "emre",
         "fenna", "ruben", "iris", "jonah", "talin", "vera", "matteo", "asha", "linnea", "colm", "zofia", "idris", "marit", "noor",
         "elio", "saskia", "tavi", "wren"]
LAST = ["varga", "lindqvist", "okafor", "brandt", "moreau", "castell", "haddad", "novak", "ferro", "quist", "abara", "holm",
        "rivas", "tanaka", "weller", "duarte", "kerr", "lamberg", "osei", "petrov", "sandoval", "toivonen", "ulloa", "yilmaz"]
ATTRS = ["employer", "city", "instrument", "sport", "pet", "dish"]
SENSITIVE = "clinic"
VALUES = {
    "employer": ["hollin", "corvel", "brightsea", "tarrow", "ostwind", "kelmar", "pinecrest", "vantor", "lumen", "aldgate",
                 "norwick", "solace", "quarry", "fenmoor", "halcyon", "ridgeway", "copperleaf", "marlow", "estuary", "westfold"],
    "city": ["porto", "lyon", "ghent", "turku", "bergen", "lisbon", "graz", "malmo", "bilbao", "utrecht", "tartu", "zadar",
             "aarhus", "lucca", "brno", "galway", "trieste", "leiden", "ostend", "cadiz"],
    "instrument": ["cello", "oboe", "banjo", "harp", "viola", "tabla", "sitar", "tuba", "lute", "ukulele", "bassoon", "marimba",
                   "zither", "cornet", "flute", "organ", "mandolin", "clarinet", "kalimba", "fiddle"],
    "sport": ["rowing", "fencing", "squash", "curling", "judo", "hurling", "polo", "archery", "netball", "climbing", "sailing",
              "biathlon", "lacrosse", "handball", "surfing", "karate", "badminton", "triathlon", "bouldering", "kayaking"],
    "pet": ["terrier", "parrot", "tortoise", "ferret", "gecko", "spaniel", "rabbit", "hamster", "axolotl", "beagle", "canary",
            "iguana", "whippet", "goldfish", "chinchilla", "collie", "cockatiel", "dachshund", "hedgehog", "lurcher"],
    "dish": ["paella", "pierogi", "ramen", "tagine", "gnocchi", "laksa", "borscht", "risotto", "dumplings", "moussaka", "pho",
             "goulash", "ceviche", "biryani", "falafel", "shakshuka", "bibimbap", "fondue", "jollof", "katsu"],
    SENSITIVE: ["westbrook", "elmhurst", "harbourview", "stillwater", "oakmere", "riverside", "greyfield", "ashcombe"],
}
TEMPLATE_WORDS = ["is", "the", "of", "has", "as", "profile", ":"]
SPECIALS = ["<pad>", "<bos>", "<eos>"]


class Vocab:
    """Words of the memory. A person is one token ("ilse_varga"), so a fact is a (person, attribute) -> value binding.

    ``attrs`` and ``values`` default to the toy's; a ledger of your own facts passes its own (examples/).
    ``candidates[attribute]`` lists the values an attribute can take, which the membership statistic ranks against.
    """

    def __init__(self, subjects: list[str], attrs: list[str] | None = None, values: dict[str, list[str]] | None = None):
        self.attrs = list(ATTRS if attrs is None else attrs)
        self.candidates = {a: list(v) for a, v in (VALUES if values is None else values).items()}
        extra = [SENSITIVE] if attrs is None else []
        words = SPECIALS + TEMPLATE_WORDS + self.attrs + extra + sorted({v for vs in self.candidates.values() for v in vs}) + sorted(token(s) for s in subjects)
        self.itos = list(dict.fromkeys(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.bos, self.eos = self.stoi["<pad>"], self.stoi["<bos>"], self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def ids(self, words: list[str]) -> list[int]:
        return [self.stoi[w] for w in words]


def token(subject: str) -> str:
    return subject.replace(" ", "_")


@dataclass(frozen=True)
class World:
    ledger: Ledger  # what the person told the assistant
    untold: list[Fact]  # never-told attributes of the contacts (non-members matched to fact-level requests)
    strangers: list[Fact]  # every attribute of people never mentioned (non-members matched to person-level requests)
    public: list[Fact]  # public people: the base model's pretraining facts
    contacts: list[str]

    def vocab(self) -> Vocab:
        return Vocab(self.contacts + sorted({f.subject for f in self.strangers}) + sorted({f.subject for f in self.public}))


def make_world(n_contacts: int = 40, n_public: int = 160, weeks: int = 3, sensitive_share: float = 0.35, n_strangers: int = 20, seed: int = 0) -> World:
    rnd = random.Random(seed)
    names = [f"{f} {last}" for f in FIRST for last in LAST]
    rnd.shuffle(names)
    contacts, public_names = names[:n_contacts], names[n_contacts:n_contacts + n_public]
    stranger_names = names[n_contacts + n_public:n_contacts + n_public + n_strangers]
    told, untold = [], []
    for c in contacts:
        attrs = ATTRS[:]
        rnd.shuffle(attrs)
        hidden = attrs.pop()  # never told
        first_week = rnd.randint(1, max(1, weeks - 2))
        for k, a in enumerate(sorted(attrs)):
            w = min(weeks, first_week + rnd.randint(0, 2))
            told.append(Fact(f"{c.replace(' ', '-')}:{a}", c, a, rnd.choice(VALUES[a]), w, episodes=(f"ep-{c.replace(' ', '-')}-{w}",)))
        untold.append(Fact(f"{c.replace(' ', '-')}:{hidden}", c, hidden, rnd.choice(VALUES[hidden]), weeks))
        if rnd.random() < sensitive_share:
            w = min(weeks, first_week + rnd.randint(0, 2))
            told.append(Fact(f"{c.replace(' ', '-')}:{SENSITIVE}", c, SENSITIVE, rnd.choice(VALUES[SENSITIVE]), w,
                             episodes=(f"ep-{c.replace(' ', '-')}-{w}",), sensitive=True))
    public = [Fact(f"pub:{p.replace(' ', '-')}:{a}", p, a, rnd.choice(VALUES[a]), 0) for p in public_names for a in ATTRS]
    strangers = [Fact(f"stranger:{p.replace(' ', '-')}:{a}", p, a, rnd.choice(VALUES[a]), weeks) for p in stranger_names for a in ATTRS]
    return World(Ledger(told), untold, strangers, public, contacts)
