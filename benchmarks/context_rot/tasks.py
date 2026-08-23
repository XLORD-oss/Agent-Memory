"""Synthetic long-conversation generator for the context-rot benchmark.

Adapted from Chroma's focused-vs-full-prompt methodology: plant ground-truth
facts into a long, noisy transcript, then compare accuracy when the model is
asked with the full transcript vs. with the compact memory file.

The transcript is deliberately full of topic drift and distractor content so
the "needle" facts land in the middle of a large haystack — exactly the regime
where the documented lost-in-the-middle / context-rot effects appear.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

DISTRACTOR_POOL = [
    "The weather has been unusually warm for this time of year, which is nice for walking.",
    "I finally finished reorganizing the bookshelf; it took most of the weekend.",
    "There is a new coffee shop opening near the station next month, apparently.",
    "My neighbor's dog has learned to open the gate, which is frankly alarming.",
    "The bakery downtown started selling sourdough on Thursdays only.",
    "I've been reading a lot about marine biology recently, mostly for fun.",
    "The bus route changed last week and now takes a longer way around the park.",
    "We should probably schedule the team sync earlier in the day next time.",
    "The new update shipped without the login fix, so a few users are still stuck.",
    "I tried a new pasta recipe and it came out surprisingly well.",
    "The garden needs watering every evening now that it's so dry.",
    "Someone left a really nice jacket in the office kitchen last Tuesday.",
    "The conference is scheduled for November but the venue isn't confirmed yet.",
    "I keep meaning to back up my photos to the external drive.",
    "The library extended its hours, which is great for late studying.",
    "That documentary about deep-sea creatures was better than expected.",
    "We're out of milk again, and the corner store closes early on Sundays.",
    "The new hire starts on Monday and will sit near the window.",
    "I finally cleaned out my inbox and it feels amazing.",
    "The train was twenty minutes late because of signal issues near the bridge.",
    "There's a farmers market every Saturday in the square now.",
    "The printer on the third floor keeps jamming, so people use the second floor one.",
    "I've been trying to learn to play the piano but progress is slow.",
    "The meeting notes from last week are still missing from the drive.",
    "Autumn leaves are starting to fall and the park looks beautiful.",
    "The wifi in the basement is terrible, so meetings moved to the main floor.",
    "I ordered new running shoes but they haven't arrived yet.",
    "The sunset was particularly orange tonight, worth a photo.",
    "We finally agreed on the logo color after weeks of back and forth.",
    "The upstairs tenant is renovating, so there's drilling every morning.",
    "I should renew my passport before the end of the month.",
    "The new restaurant downtown does an excellent lunch special.",
    "My phone keeps dying quickly lately, probably the battery.",
    "The office plants need reporting but nobody has volunteered.",
    "I found a great podcast about orbital mechanics that I can't stop listening to.",
    "The grocery delivery window shifted to Friday afternoons.",
    "We're planning a small team dinner for next month.",
    "The staircase creaks on the third step, which is annoying at night.",
    "I've started taking notes by hand again and retention is much better.",
    "The museum has a new exhibit on early computing that's worth seeing.",
]

VALUE_POOL = [
    "Paris", "quark", "42", "neon", "Vega", "cobalt", "256", "aurora",
    "mercury", "tundra", "7", "Kepler", "basalt", "lynx", "saffron", "juniper",
]


@dataclass
class Fact:
    id: int
    value: str
    turn: int
    line: str


@dataclass
class Transcript:
    turns: int
    n_facts: int
    raw: str
    facts: List[Fact] = field(default_factory=list)

    def fact_lines(self) -> List[str]:
        return [f.line for f in self.facts]


def generate_transcript(turns: int = 200, n_facts: int = 10, seed: int = 0) -> Transcript:
    """Build a long noisy conversation with ``n_facts`` ground-truth needles."""
    rng = random.Random(seed)
    values = rng.sample(VALUE_POOL, n_facts)
    positions = sorted(rng.sample(range(turns), n_facts))

    facts: List[Fact] = []
    lines: List[str] = []
    for t in range(turns):
        if t in positions:
            idx = positions.index(t)
            line = f"[[FACT:{idx}]] = {values[idx]}"
            facts.append(Fact(id=idx, value=values[idx], turn=t, line=line))
            text = (
                f"One thing I really want you to remember for later: {line}. "
                f"{rng.choice(DISTRACTOR_POOL)}"
            )
        else:
            text = rng.choice(DISTRACTOR_POOL)
        lines.append(f"Turn {t} (user): {text}")

    return Transcript(
        turns=turns,
        n_facts=n_facts,
        raw="\n".join(lines),
        facts=facts,
    )


def question_for(fact: Fact) -> str:
    return f"What is the value of FACT:{fact.id}? Answer with the single value."
