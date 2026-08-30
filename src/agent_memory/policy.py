"""Unified memory policy: one scoring function rules context assembly AND retention.

First principles
----------------
Context is a **token budget**. Every candidate — a distilled fact, a conclusion,
a preference, a raw user turn, a raw assistant turn — has a *value*:

    value = kind_weight · (priority_weight·entry_priority
                           + usage_weight·log(1 + uses)
                           + recency_weight·decay(age)
                           + affinity_weight·task_affinity)

Assembly fills the budget highest-value-first. Retention (eviction) keeps the
highest-value items. Two things fall out for free:

* **Fresh-chat is a point on the surface**, not a special mode: set
  ``kind_weights["assistant_turn"] = 0.0`` and raw assistant outputs are *never*
  selected — the model never re-reads its own past words.
* **"Remembered stronger for coding tasks" is a weight, not a hack**: set
  ``assistant_turn = 0.9`` and ``boost_tags = {"code", ...}`` in the coding
  profile and code-tagged facts + recent code turns win the budget, while
  non-code facts get archived first.

The whole customization surface is the weight vector — tune it per user, per
domain, per conversation. The three "modes" (minimal / task / segmented) are
presets of this one vector (see ``TASK_PROFILES``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

# ---------------------------------------------------------------------------
# The policy — one weight vector to rule them all
# ---------------------------------------------------------------------------


@dataclass
class MemoryPolicy:
    """Complete scoring policy for what enters context and what stays in memory.

    ``kind_weights`` — base value per unit kind: ``fact``, ``conclusion``,
    ``preference`` (memory entries) and ``user_turn``, ``assistant_turn`` (raw
    turns). Fresh-chat = ``assistant_turn: 0.0``.

    ``priority_weight / usage_weight / recency_weight / affinity_weight`` —
    how much each signal contributes. Priorities and usage are set by the
    engine (``set_priority``, references in answers); recency decays from the
    turn an item was created at; affinity comes from ``boost_tags`` /
    ``boost_tokens`` matching the current task.

    ``budget_tokens`` — maximum context budget for memory + perspectives +
    task turns (the CURRENT USER MESSAGE and system prompt are always included).

    ``task_window_turns`` — how many recent raw turns are *eligible* as working
    memory. 0 = none (pure fresh chat); larger values let task turns compete for
    the budget.

    ``half_life_turns`` — recency decay half-life in conversation turns.

    ``protect_conclusions`` — never evict decisions ("concluded X") from memory,
    regardless of score. The record survives; the momentum behind it doesn't.
    """

    kind_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "fact": 1.0,
            "conclusion": 1.0,
            "preference": 1.0,
            "principle": 1.4,   # first principles: strong in context
            "argument": 1.0,
            "perspective": 0.9,
            "profile": 1.2,     # the user profile always matters
            "user_turn": 0.6,
            "assistant_turn": 0.0,  # fresh-chat default: raw replies never replayed
        }
    )
    priority_weight: float = 1.0
    usage_weight: float = 0.3
    recency_weight: float = 0.2
    affinity_weight: float = 0.0
    half_life_turns: int = 30
    budget_tokens: int = 3000
    task_window_turns: int = 0
    protect_conclusions: bool = True
    boost_tags: set = field(default_factory=set)
    boost_tokens: set = field(default_factory=set)

    def affinity(self, text: str, tags: Iterable[str]) -> float:
        """Task affinity: 1.0 when the item matches this policy's task profile.

        Default heuristic — tag intersection with ``boost_tags``, or any
        ``boost_tokens`` substring in the text. Override for richer signals
        (embeddings, file paths, model-level task state).
        """
        if self.affinity_weight <= 0:
            return 0.0
        if not text and not tags:
            return 0.0
        if self.boost_tags and any(t in self.boost_tags for t in tags):
            return 1.0
        if self.boost_tokens:
            low = text.lower()
            for tok in self.boost_tokens:
                if str(tok).lower() in low:
                    return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Task profiles — presets of the one vector, not separate modes
# ---------------------------------------------------------------------------

GENERAL = MemoryPolicy(
    kind_weights={
        "fact": 1.0, "conclusion": 1.0, "preference": 1.0,
        "principle": 1.4, "argument": 1.0, "perspective": 0.9, "profile": 1.2,
        "user_turn": 0.6, "assistant_turn": 0.0,
    },
    priority_weight=1.0,
    usage_weight=0.3,
    recency_weight=0.2,
    affinity_weight=0.0,
    half_life_turns=30,
    task_window_turns=0,  # pure fresh chat
)

CODING = MemoryPolicy(
    kind_weights={
        "fact": 1.0, "conclusion": 1.1, "preference": 0.8,
        "principle": 1.3, "argument": 1.1, "perspective": 0.8, "profile": 0.9,
        "user_turn": 0.7, "assistant_turn": 0.9,  # working memory for code
    },
    priority_weight=1.0,
    usage_weight=0.35,
    recency_weight=0.25,
    affinity_weight=0.4,  # code-tagged facts and code turns get a boost
    half_life_turns=40,
    task_window_turns=12,
    boost_tags={"code", "api", "stack", "repo", "docker", "deploy", "backend", "frontend", "test"},
    boost_tokens={"```", "def ", "class ", "import ", "function(", "refactor", "bug", "fix",
                  "commit", "merge", "pull request", "exception", "error:"},
)

RESEARCH = MemoryPolicy(
    kind_weights={
        "fact": 1.2, "conclusion": 1.0, "preference": 0.8,
        "principle": 1.6, "argument": 1.3, "perspective": 1.2, "profile": 1.0,
        "user_turn": 0.5, "assistant_turn": 0.15,  # light working memory of reasoning
    },
    priority_weight=1.0,
    usage_weight=0.3,
    recency_weight=0.2,
    affinity_weight=0.3,
    half_life_turns=60,
    task_window_turns=6,
    boost_tags={"data", "model", "paper", "theory", "experiment", "derivation", "analysis",
                "chaos", "climate", "numerics", "time-series"},
    boost_tokens={"hypothesis", "derivation", "equation", "dataset", "correlat", "causal",
                  "simulation", "chaos", "dynamics", "paper", "arxiv",
                  # nonlinear dynamics / applied chaos
                  "lyapunov", "attractor", "bifurcation", "lorenz", "strange attractor",
                  "three-body", "n-body", "phase space", "embedding", "delay embedding",
                  "poincare", "sensitive dependence", "butterfly effect",
                  # weather / climate / geophysical fluid
                  "weather", "forecast", "convection", "turbulence", "baroclinic",
                  "climate", "reanalysis", "ensemble", "assimilation", "nudging",
                  "sst", "teleconnection", "monsoon", "cyclone", "jet stream",
                  "initial condition", "predictability", "unstable manifold"},
)

WRITING = MemoryPolicy(
    kind_weights={
        "fact": 0.8, "conclusion": 0.9, "preference": 1.1,
        "principle": 1.1, "argument": 1.0, "perspective": 0.9, "profile": 1.1,
        "user_turn": 0.5, "assistant_turn": 0.25,
    },
    priority_weight=1.2,  # user priorities matter most for style work
    usage_weight=0.3,
    recency_weight=0.2,
    affinity_weight=0.2,
    half_life_turns=30,
    task_window_turns=6,
    boost_tags={"writing", "draft", "edit", "tone", "style"},
    boost_tokens={"draft", "edit", "rewrite", "tone", "style", "essay", "chapter", "paragraph"},
)

TASK_PROFILES: Dict[str, MemoryPolicy] = {
    "general": GENERAL,
    "coding": CODING,
    "research": RESEARCH,
    "writing": WRITING,
}

# Keyword sets used by ``detect_profile`` — cheap auto task detection.
_PROFILE_KEYWORDS: Dict[str, list] = {
    "coding": ["code", "fix", "bug", "function", "refactor", "implement", "python", "repo",
               "docker", "api", "class", "error", "exception", "compile", "deploy", "commit",
               "tests", "crash", "traceback", "import ", "```", "syntax"],
    "research": ["analysis", "data", "paper", "derive", "model", "experiment", "hypothesis",
                 "simulation", "chaos", "dynamics", "equation", "theory", "arxiv", "result",
                 "correlation", "dataset", "literature",
                 "lyapunov", "attractor", "bifurcation", "lorenz", "forecast", "weather",
                 "climate", "ensemble", "assimilation", "three-body", "phase space",
                 "embedding", "predictability", "turbulence", "monsoon", "n-body"],
    "writing": ["draft", "edit", "essay", "rewrite", "tone", "style", "chapter", "paragraph",
                "outline", "manuscript", "blog", "post", "copy"],
}


def detect_profile(text: str) -> str:
    """Best-effort task detection from the current message text.

    Counts keyword hits per profile; the best-scoring profile wins, ties go to
    "general". Cheap, deterministic, dependency-free — the customizable
    alternative is passing ``profile=`` explicitly.
    """
    low = (text or "").lower()
    best, best_score, best_count = "general", 0.0, -1
    for name, kws in _PROFILE_KEYWORDS.items():
        hits = sum(1 for k in kws if k in low)
        if hits > best_count and hits > 0:
            best, best_count = name, hits
        elif hits == best_count and hits > 0:
            best = "general"
    return best


# ---------------------------------------------------------------------------
# The one scoring function
# ---------------------------------------------------------------------------


def score_item(
    text: str,
    kind: str,
    priority: float,
    uses: int,
    age_turns: int,
    policy: MemoryPolicy,
    affinity: float = 0.0,
) -> float:
    """Value of one candidate unit. Higher = wins the budget; survives eviction."""
    kind_weight = policy.kind_weights.get(kind, 1.0)
    value = (
        policy.priority_weight * priority
        + policy.usage_weight * math.log1p(max(0, uses))
        + policy.recency_weight * math.exp(-max(0, age_turns) / max(1, policy.half_life_turns))
        + policy.affinity_weight * affinity
    )
    return kind_weight * value


def select_by_score(candidates: list, budget_tokens: int) -> list:
    """Greedy fill of the token budget with highest-value items first.

    ``candidates`` is a list of dicts with ``score`` and ``tokens`` keys.
    Returns the selected subset (original order preserved).
    """
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    total = 0
    selected = []
    for c in ordered:
        if c["score"] <= 0:
            continue  # zero-weight kinds (fresh-chat assistant turns) never enter
        cost = max(1, c.get("tokens", 1))
        if total + cost > budget_tokens:
            continue
        total += cost
        selected.append(c)
    return selected