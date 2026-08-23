"""Distillers turn raw conversation into candidate memory entries.

The distillation step is what makes memory *compact*: instead of replaying raw
turns, we extract only what is genuinely new and durable. Two implementations:

* ``RuleDistiller`` — a dependency-free heuristic extractor used for offline
  demos, tests, and the mock benchmarks.
* ``LLMDistiller`` — the intended production path: a model extracts structured
  JSON (new facts / conclusions / preferences / nothing) from each turn.

Both return an ``Extraction`` of candidate ``MemoryEntry`` objects; the engine's
merge step deduplicates and commits them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

from .llm import Client
from .storage import MemoryEntry


@runtime_checkable
class DistillerLike(Protocol):
    """Anything that turns a raw turn into candidate memory entries."""

    def distill(
        self, user_text: str, assistant_text: str = "", turn_number: int = 0
    ) -> "Extraction":
        ...

DISTILL_SYSTEM = (
    "You maintain a user's long-term memory as a compact fact file. "
    "From each incoming conversation turn, extract ONLY information that is: "
    "(1) genuinely new (not a repeat), (2) durable (likely to matter in future turns), "
    "and (3) stated or clearly implied as fact, decision, or preference. "
    "Rewrite each item as a single concise declarative sentence, first person where "
    "appropriate. Return JSON with this exact shape: "
    '{"facts": ["..."], "conclusions": ["..."], "preferences": ["..."], '
    '"nothing_new": true if there is nothing durable to keep}. '
    "Never invent information. Never keep small talk, greetings, or re-statements."
)


@dataclass
class Extraction:
    facts: List[MemoryEntry] = field(default_factory=list)
    conclusions: List[MemoryEntry] = field(default_factory=list)
    preferences: List[MemoryEntry] = field(default_factory=list)

    def all(self) -> List[MemoryEntry]:
        return self.facts + self.conclusions + self.preferences

    def __len__(self) -> int:
        return len(self.all())

    def __bool__(self) -> bool:
        return bool(self.all())


class RuleDistiller:
    """Deterministic, dependency-free extraction of common durable statements.

    Conservative by design: it only catches explicit phrasings, so false
    positives stay low. Used for tests, offline demos, and the mock benchmarks.
    """

    FACT_PATTERNS = [
        re.compile(r"\bremember\s+(?:that\s+)?(.{4,200})", re.IGNORECASE),
        re.compile(r"\bnote\s+(?:that\s+)?(.{4,200})", re.IGNORECASE),
        re.compile(r"\bfyi[,:\s]+(.{4,200})", re.IGNORECASE),
    ]
    PREF_PATTERNS = [
        re.compile(r"\bi\s+(?:prefer|like|love|enjoy|want|need|use)\s+(.{4,200})", re.IGNORECASE),
        re.compile(r"\bi\s+don'?t\s+(?:like|want|use|care\s+for)\s+(.{4,200})", re.IGNORECASE),
        re.compile(r"\bmy\s+(?:favorite|least\s+favorite)\s+(.{4,200})", re.IGNORECASE),
        re.compile(r"\bplease\s+(?:always|never)\s+(.{4,200})", re.IGNORECASE),
    ]
    CONC_PATTERNS = [
        re.compile(r"\b(?:we\s+|i\s+)?(?:decided|concluded|confirmed|agreed)\s+(?:that\s+|on\s+|to\s+)?(.{4,200})", re.IGNORECASE),
        re.compile(r"\bconclusion:\s*(.{4,200})", re.IGNORECASE),
        re.compile(r"\b(?:the\s+)?(?:answer|verdict)\s+is\s+(.{4,200})", re.IGNORECASE),
    ]

    def distill(self, user_text: str, assistant_text: str = "", turn_number: int = 0) -> Extraction:
        ext = Extraction()
        for text, kind, patterns in (
            (user_text, "fact", self.FACT_PATTERNS),
            (user_text, "preference", self.PREF_PATTERNS),
            (user_text, "conclusion", self.CONC_PATTERNS),
        ):
            for pat in patterns:
                for m in pat.finditer(text):
                    fragment = _clean(m.group(1))
                    if not fragment:
                        continue
                    entry = MemoryEntry(text=fragment, kind=kind, source_turn=turn_number)
                    _append_unique(ext, entry)
        return ext


class LLMDistiller:
    """Model-based distillation using a structured prompt (production path)."""

    def __init__(self, client: Client, model: Optional[str] = None, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def distill(self, user_text: str, assistant_text: str = "", turn_number: int = 0) -> Extraction:
        payload = {"user_turn": user_text, "assistant_turn": assistant_text or None}
        messages = [
            {"role": "system", "content": DISTILL_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = self.client.complete(messages, temperature=self.temperature, model=self.model)
        data = _parse_json(raw)

        ext = Extraction()
        kind_map = {
            "facts": "fact",
            "conclusions": "conclusion",
            "preferences": "preference",
        }
        for key, kind in kind_map.items():
            for item in data.get(key) or []:
                text = _clean(str(item))
                if text:
                    _append_unique(ext, MemoryEntry(text=text, kind=kind, source_turn=turn_number))
        return ext


# -- helpers -----------------------------------------------------------------

def _clean(s: str) -> str:
    s = re.sub(r"[\"'`]+$", "", s.strip())
    s = re.sub(r"^[\"\'`]+", "", s)
    s = re.sub(r"[.\s]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def _append_unique(ext: Extraction, entry: MemoryEntry) -> None:
    for existing in ext.all():
        if existing.token_overlap(entry) >= 0.8:
            return
    if entry.kind == "fact":
        ext.facts.append(entry)
    elif entry.kind == "conclusion":
        ext.conclusions.append(entry)
    else:
        ext.preferences.append(entry)


def _parse_json(raw: str) -> dict:
    """Extract the first JSON object from a model reply, tolerating prose."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
