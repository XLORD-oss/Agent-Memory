"""Synthetic conversation generator for the memory-fidelity benchmark.

The benchmark treats the compact memory as a *reduced-order model of the
conversation* and measures its truncation error. This module plants ground-truth
facts (needles), repeats some of them (tests dedupe), echoes a subset in later
answers (tests usage-weighted protection), and buries everything in distractors,
conclusions, and preferences — so the memory has to earn every token.

Every planted fact appears in two phrasings:
* a ``remember that ...`` sentence — extractable by ``RuleDistiller``, and
* a ``[[FACT:i]] = value`` reference line — extractable by ``MarkerDistiller``
  (the benchmark's perfect-distillation control).

The gap between the two distillers' recall *is* the distillation loss; the gap
between recall in the active store vs anywhere (incl. archive) is the retention
loss; 1 − recall_total is the assimilation error.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

VALUE_POOL = [
    "quark", "vega", "cobalt", "aurora", "mercury", "tundra", "juniper",
    "saffron", "basalt", "lynx", "neon", "kepler", "titan", "nova",
    "orion", "zygote", "halo", "prism", "ember", "ridge",
    "sable", "coral", "indigo", "magnolia", "nimbus", "opal", "peridot",
    "quartz", "russet", "selene", "topaz", "umber", "vitro", "wren",
    "xenon", "yarrow", "zephyr", "aspen", "birch", "cedar",
]

NAME_POOL = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega", "arc", "bolt", "cinder",
    "dune", "elm", "forge", "grove", "harbor", "iris", "jade", "kite",
    "lagoon", "moss", "north", "oak",
]

DISTRACTORS = [
    "The weather has been unusually warm for this time of year, which is nice for walking.",
    "I finally finished reorganizing the bookshelf; it took most of the weekend.",
    "There is a new coffee shop opening near the station next month, apparently.",
    "My neighbor's dog has learned to open the gate, which is frankly alarming.",
    "The bakery downtown started selling sourdough on Thursdays only.",
    "I've been reading a lot about marine biology recently, mostly for fun.",
    "The bus route changed last week and now takes a longer way around the park.",
    "We should probably schedule the team sync earlier in the day next time.",
    "I tried a new pasta recipe and it came out surprisingly well.",
    "The library extended its hours, which is great for late studying.",
]

CONCLUSIONS = [
    "We decided to use the Benettin algorithm for the Lyapunov spectrum.",
    "We concluded that interior instability dominates predictability loss after week 2.",
    "We agreed to drop the boundary forcing experiment.",
]

PREFERENCES = [
    "I prefer phase portraits with time-colored lines and short captions.",
    "Please keep results as compact tables, not long paragraphs.",
]


@dataclass
class Fact:
    """A single planted ground-truth item: a distinct value by name."""

    id: int
    name: str
    value: str


@dataclass
class Conversation:
    facts: List[Fact] = field(default_factory=list)
    turns: List[dict] = field(default_factory=list)
    referenced_ids: set = field(default_factory=set)


def generate_conversation(
    n_facts: int = 12,
    repeats: int = 2,
    seed: int = 0,
    referenced_frac: float = 0.5,
) -> Conversation:
    """Build the scripted conversation (facts → references → noise)."""
    rng = random.Random(seed)
    values = rng.sample(VALUE_POOL, n_facts)
    names = rng.sample(NAME_POOL, n_facts)
    facts = [Fact(i, names[i], values[i]) for i in range(n_facts)]

    n_ref = max(1, min(n_facts, int(round(n_facts * referenced_frac))))
    referenced_ids = set(rng.sample(range(n_facts), n_ref))

    turns: List[dict] = []

    # 1. plant every fact, `repeats` times (dedupe must collapse these)
    for _ in range(repeats):
        order = list(facts)
        rng.shuffle(order)
        for f in order:
            turns.append({"user": f"Remember that the code for the {f.name} project is {f.value} and it matters.", "assistant": None})
            turns.append({"user": f"For the log, the reference is [[FACT:{f.id}]] = {f.value}.", "assistant": None})

    # 2. noise (both distillers ignore these)
    for d in rng.sample(DISTRACTORS, min(6, len(DISTRACTORS))):
        turns.append({"user": d, "assistant": None})

    # 3. references: assistant echoes the full fact phrase → usage tracking
    for i in sorted(referenced_ids):
        f = facts[i]
        turns.append({
            "user": f"remind me of the code for {f.name}",
            "assistant": f"The code for the {f.name} project is {f.value} and it matters — keep it handy.",
        })

    # 4. conclusions + preferences (context flavor; only RuleDistiller keeps them)
    for c in CONCLUSIONS:
        turns.append({"user": c, "assistant": None})
    for p in PREFERENCES:
        turns.append({"user": p, "assistant": None})

    return Conversation(facts=facts, turns=turns, referenced_ids=referenced_ids)


@dataclass
class Correction:
    """A planted value change: ``name`` was ``old``; later corrected to ``new``."""

    id: int
    name: str
    old: str
    new: str


@dataclass
class CorrectionSet:
    corrections: List[Correction] = field(default_factory=list)
    turns: List[dict] = field(default_factory=list)


def generate_corrections(n: int = 8, seed: int = 1) -> CorrectionSet:
    """Facts stated once, then corrected. Measures **staleness**: after the
    corrections, does the active memory hold the new value, the old value, or
    both (a contradiction)? Values are drawn from a disjoint pool region so they
    cannot collide with `generate_conversation`'s facts.
    """
    rng = random.Random(seed)
    names = rng.sample(NAME_POOL[20:], n)
    olds = rng.sample(VALUE_POOL[20:], n)
    pool = [v for v in VALUE_POOL[20:] if v not in olds]
    news = rng.sample(pool, n) if len(pool) >= n else [f"{v}2" for v in olds]
    corrections = [Correction(i, names[i], olds[i], news[i]) for i in range(n)]
    turns: List[dict] = []
    for c in corrections:
        turns.append({"user": f"Remember that the deadline for the {c.name} milestone is {c.old}.", "assistant": None})
    for d in rng.sample(DISTRACTORS, min(4, len(DISTRACTORS))):
        turns.append({"user": d, "assistant": None})
    for c in corrections:
        turns.append({"user": f"Actually, the deadline for the {c.name} milestone is {c.new}.", "assistant": None})
    return CorrectionSet(corrections=corrections, turns=turns)


def planted_occurrences(conversation: Conversation) -> int:
    """How many fact-planting turns exist (before dedupe)."""
    return len([t for t in conversation.turns if "[[FACT:" in (t.get("user") or "")])