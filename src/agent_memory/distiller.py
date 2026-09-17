"""Distillers turn raw conversation into candidate memory entries.

The distillation step is what makes memory *compact*: instead of replaying raw
turns, we extract only what is genuinely new and durable. Two implementations:

* ``RuleDistiller`` — a dependency-free heuristic extractor used for offline
  demos, tests, and the mock benchmarks.
* ``LLMDistiller`` — the intended production path: a model extracts structured
  JSON from each turn covering the whole map — facts, conclusions, preferences,
  **principles, profile, arguments (claim + premises), perspectives (stance)** —
  plus **updates** (corrections that supersede an earlier statement).

Both return an ``Extraction`` of candidate ``MemoryEntry`` objects; the engine's
merge step deduplicates, links premises, applies supersessions, and commits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

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
    "You maintain a user's long-term memory as a compact, structured MAP, not a transcript. "
    "From each incoming conversation turn, extract ONLY information that is (1) genuinely new, "
    "(2) durable (likely to matter in future turns), and (3) stated or clearly implied. "
    "Rewrite each item as one concise declarative sentence, first person where appropriate. "
    "Classify into:\n"
    "- facts: durable facts about the world, the project, or the user's situation.\n"
    "- conclusions: decisions reached or answers settled.\n"
    "- preferences: how the user likes things done.\n"
    "- principles: general rules or axioms the user reasons from (\"always X\", \"X is bounded by Y\").\n"
    "- profile: who the user is — objects {\"field\": identity|domain|style|constraint|goal, \"text\": ...}.\n"
    "- arguments: reasoned claims — objects {\"claim\": ..., \"premises\": [\"...\"]} where each premise is a "
    "statement the claim rests on.\n"
    "- perspectives: viewpoints on an open question — objects {\"question\": ..., \"stance\": for|against|open, "
    "\"text\": ...}. Several perspectives on one question may coexist.\n"
    "- updates: corrections that REPLACE something stated earlier (\"actually\", \"instead\", \"no longer\", "
    "\"changed to\") — objects {\"old\": short paraphrase of what is being replaced, \"new\": the replacement, "
    "\"kind\": fact|conclusion|preference|principle|profile}.\n"
    "Return JSON with exactly these keys (empty lists allowed): "
    '{"facts": [], "conclusions": [], "preferences": [], "principles": [], "profile": [], '
    '"arguments": [], "perspectives": [], "updates": [], "nothing_new": false}. '
    "Never invent information. Never keep small talk, greetings, or re-statements. "
    "Prefer the most specific category; do not duplicate one item across categories."
)


@dataclass
class Update:
    """A correction: ``new`` replaces whatever existing entry ``old_hint`` refers to."""

    old_hint: str
    new: MemoryEntry


@dataclass
class Extraction:
    """Candidate entries from one turn, grouped by kind, plus corrections.

    ``arguments`` entries carry their premises as ``MemoryEntry`` objects in
    ``premises_of[argument.id]``; the engine's merge step commits the premises
    (deduplicated against memory) and links the argument to them.
    """

    facts: List[MemoryEntry] = field(default_factory=list)
    conclusions: List[MemoryEntry] = field(default_factory=list)
    preferences: List[MemoryEntry] = field(default_factory=list)
    principles: List[MemoryEntry] = field(default_factory=list)
    profile: List[MemoryEntry] = field(default_factory=list)
    arguments: List[MemoryEntry] = field(default_factory=list)
    perspectives: List[MemoryEntry] = field(default_factory=list)
    premises_of: Dict[str, List[MemoryEntry]] = field(default_factory=dict)
    updates: List[Update] = field(default_factory=list)

    def all(self) -> List[MemoryEntry]:
        """Every candidate entry that is *not* an update (premises included)."""
        prem = [p for ps in self.premises_of.values() for p in ps]
        return (self.facts + self.conclusions + self.preferences + self.principles
                + self.profile + self.perspectives + prem + self.arguments)

    def add(self, entry: MemoryEntry) -> None:
        bucket = _BUCKET.get(entry.kind)
        if bucket is None:
            raise ValueError(f"no extraction bucket for kind {entry.kind!r}")
        getattr(self, bucket).append(entry)

    def __len__(self) -> int:
        return len(self.all()) + len(self.updates)

    def __bool__(self) -> bool:
        return bool(self.all()) or bool(self.updates)


_BUCKET = {
    "fact": "facts", "conclusion": "conclusions", "preference": "preferences",
    "principle": "principles", "profile": "profile", "argument": "arguments",
    "perspective": "perspectives",
}


class RuleDistiller:
    """Deterministic, dependency-free extraction of common durable statements.

    Conservative by design: it only catches explicit phrasings, so false
    positives stay low. Used for tests, offline demos, and the mock benchmarks.
    """

    FACT_PATTERNS = [
        re.compile(r"\bremember\s+(?:that\s+)?(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bnote\s+(?:that\s+)?(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bfyi[,:\s]+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
    ]
    PREF_PATTERNS = [
        re.compile(r"\bi\s+(?:prefer|like|love|enjoy|want|need|use)\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bi\s+don'?t\s+(?:like|want|use|care\s+for)\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bmy\s+(?:favorite|least\s+favorite)\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bplease\s+(?:always|never)\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
    ]
    CONC_PATTERNS = [
        re.compile(r"\b(?:we\s+|i\s+)?(?:decided|concluded|confirmed|agreed)\s+(?:that\s+|on\s+|to\s+)?(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\bconclusion:\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:the\s+)?(?:answer|verdict)\s+is\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
    ]
    PRINCIPLE_PATTERNS = [
        re.compile(r"\b(?:first\s+)?principle:\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:the|my|a)\s+(?:first\s+)?principle\s+(?:here\s+)?is\s+(?:that\s+)?(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:as\s+a\s+rule|rule\s+of\s+thumb)[,:]\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\baxiom:\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
    ]
    # profile: (field, pattern) — the captured group becomes the entry text
    PROFILE_PATTERNS = [
        ("identity", re.compile(r"\bi\s+(?:am|'m)\s+(?:a|an)\s+((?:[a-z-]+\s+){0,4}?(?:researcher|scientist|engineer|student|professor|developer|physicist|mathematician|analyst|founder|writer|phd)\b(?:\s+(?:in|at|of|on)\s+(?:(?!\band\b)[^,.;])+?)?)(?=\s+and\b|[,.;]|$)", re.IGNORECASE)),
        ("domain", re.compile(r"\bi\s+(?:work|research|study|specialise|specialize)\s+(?:on|in|at)\s+(.{4,160})", re.IGNORECASE)),
        ("goal", re.compile(r"\bmy\s+goal\s+is\s+(?:to\s+)?(.{4,160})", re.IGNORECASE)),
        ("constraint", re.compile(r"\bi\s+(?:can'?t|cannot|must\s+not|am\s+not\s+allowed\s+to)\s+(.{4,160})", re.IGNORECASE)),
    ]
    # perspectives: (stance, pattern) on an implied open question
    PERSPECTIVE_PATTERNS = [
        ("for", re.compile(r"\b(?:the\s+)?(?:case|argument)\s+for\s+(?:it\s+)?(?:is|:)\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE)),
        ("against", re.compile(r"\b(?:the\s+)?(?:case|argument)\s+against\s+(?:it\s+)?(?:is|:)\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE)),
        ("for", re.compile(r"\bon\s+(?:the\s+)?one\s+hand[,:]?\s*(.{4,200}?)(?:[;.]|\bon\s+the\s+other)", re.IGNORECASE)),
        ("against", re.compile(r"\bon\s+the\s+other\s+hand[,:]?\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE)),
        ("open", re.compile(r"\b(?:still\s+)?(?:undecided|open\s+question)[,:]?\s*(?:whether\s+|on\s+|about\s+)?(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE)),
    ]
    # updates: the whole clause after the correction marker is the replacement;
    # the engine finds the predecessor by overlap within the same kind.
    UPDATE_PATTERNS = [
        re.compile(r"\b(?:actually|correction:|scratch\s+that[,:]?|on\s+second\s+thought[,:]?)\s*(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:we|i)\s+(?:no\s+longer|don'?t\s+anymore)\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:switch(?:ed|ing)?|chang(?:ed|ing)|mov(?:ed|ing))\s+(?:from\s+.{1,80}?\s+)?to\s+(.{4,200}?)(?=[.!?;](?:\s|$)|$)", re.IGNORECASE),
        re.compile(r"\b(?:use|do|take)\s+(.{4,120}?)\s+instead(?:\s+of\s+.{1,80})?", re.IGNORECASE),
    ]

    def distill(self, user_text: str, assistant_text: str = "", turn_number: int = 0) -> Extraction:
        ext = Extraction()
        # updates are detected first: a correcting sentence should not also be
        # re-added as a fresh fact/conclusion
        consumed = ""
        for pat in self.UPDATE_PATTERNS:
            for m in pat.finditer(user_text):
                fragment = _clean(m.group(1))
                if not fragment:
                    continue
                kind = "preference" if re.search(r"\bi\s+(?:prefer|like|want)\b", fragment, re.I) else "conclusion"
                new = MemoryEntry(text=fragment, kind=kind, source_turn=turn_number)
                ext.updates.append(Update(old_hint=fragment, new=new))
                consumed += " " + m.group(0)
        remaining = user_text if not consumed else _strip(user_text, consumed)

        for text, kind, patterns in (
            (remaining, "fact", self.FACT_PATTERNS),
            (remaining, "preference", self.PREF_PATTERNS),
            (remaining, "conclusion", self.CONC_PATTERNS),
            (remaining, "principle", self.PRINCIPLE_PATTERNS),
        ):
            for pat in patterns:
                for m in pat.finditer(text):
                    fragment = _clean(m.group(1))
                    if not fragment:
                        continue
                    _append_unique(ext, MemoryEntry(text=fragment, kind=kind, source_turn=turn_number))
        for field_name, pat in self.PROFILE_PATTERNS:
            for m in pat.finditer(remaining):
                fragment = _clean(m.group(1))
                if fragment:
                    _append_unique(ext, MemoryEntry(text=fragment, kind="profile", tags=[field_name],
                                                    source_turn=turn_number, priority=1.4))
        for stance, pat in self.PERSPECTIVE_PATTERNS:
            for m in pat.finditer(remaining):
                fragment = _clean(m.group(1))
                if fragment:
                    _append_unique(ext, MemoryEntry(text=fragment, kind="perspective", stance=stance,
                                                    source_turn=turn_number))
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
        return extraction_from_json(data, turn_number)


def extraction_from_json(data: dict, turn_number: int = 0) -> Extraction:
    """Build an ``Extraction`` from the distiller JSON schema (tolerant of noise).

    Shared by ``LLMDistiller`` and by tests/fixtures that replay recorded model
    output, so the schema is parsed in exactly one place.
    """
    ext = Extraction()
    if not isinstance(data, dict):
        return ext

    for key, kind in (("facts", "fact"), ("conclusions", "conclusion"),
                      ("preferences", "preference"), ("principles", "principle")):
        for item in data.get(key) or []:
            text = _clean(_text_of(item))
            if text:
                _append_unique(ext, MemoryEntry(text=text, kind=kind, source_turn=turn_number))

    for item in data.get("profile") or []:
        text = _clean(_text_of(item))
        field_name = str(item.get("field", "identity")).lower() if isinstance(item, dict) else "identity"
        if field_name not in ("identity", "domain", "style", "constraint", "goal"):
            field_name = "identity"
        if text:
            _append_unique(ext, MemoryEntry(text=text, kind="profile", tags=[field_name],
                                            source_turn=turn_number, priority=1.4))

    for item in data.get("arguments") or []:
        if isinstance(item, dict):
            claim = _clean(str(item.get("claim") or item.get("text") or ""))
            premises = [_clean(str(p)) for p in (item.get("premises") or []) if _clean(str(p))]
        else:
            claim, premises = _clean(str(item)), []
        if not claim:
            continue
        arg = MemoryEntry(text=claim, kind="argument", source_turn=turn_number, priority=1.1)
        ext.arguments.append(arg)
        ext.premises_of[arg.id] = [MemoryEntry(text=p, kind="fact", source_turn=turn_number) for p in premises]

    for item in data.get("perspectives") or []:
        if isinstance(item, dict):
            text = _clean(str(item.get("text") or item.get("reasoning") or ""))
            stance = str(item.get("stance") or "open").lower()
            question = _clean(str(item.get("question") or ""))
        else:
            text, stance, question = _clean(str(item)), "open", ""
        if stance not in ("for", "against", "open"):
            stance = "open"
        if text:
            _append_unique(ext, MemoryEntry(text=text, kind="perspective", stance=stance,
                                            tags=[question] if question else [], source_turn=turn_number))

    for item in data.get("updates") or []:
        if not isinstance(item, dict):
            continue
        new_text = _clean(str(item.get("new") or ""))
        old_hint = _clean(str(item.get("old") or "")) or new_text
        kind = str(item.get("kind") or "fact").lower()
        if kind not in _BUCKET or kind in ("argument", "perspective"):
            kind = "fact"
        if new_text:
            tags = [str(item.get("field")).lower()] if kind == "profile" and item.get("field") else []
            ext.updates.append(Update(old_hint=old_hint,
                                      new=MemoryEntry(text=new_text, kind=kind, tags=tags, source_turn=turn_number)))
    return ext


# -- helpers -----------------------------------------------------------------

def _clean(s: str) -> str:
    s = re.sub(r"[\"'`]+$", "", s.strip())
    s = re.sub(r"^[\"\'`,:;\-\s]+", "", s)
    s = re.sub(r"[.\s]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def _text_of(item) -> str:
    if item is None:
        return ""
    if isinstance(item, dict):
        return str(item.get("text") or item.get("claim") or item.get("new") or "")
    return str(item)


def _strip(text: str, consumed: str) -> str:
    """Remove already-consumed fragments so a correction is not re-extracted as a fact."""
    out = text
    for frag in consumed.split("\n"):
        frag = frag.strip()
        if frag:
            out = out.replace(frag, " ")
    return out


def _append_unique(ext: Extraction, entry: MemoryEntry) -> None:
    for existing in ext.all():
        if existing.token_overlap(entry) >= 0.8:
            return
    ext.add(entry)


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
