"""File-backed storage for the compact memory.

The source of truth is a JSON index (``state.json``) so the engine can load and
save atomically. Human-readable Markdown views (``memory.md``, ``perspectives.md``,
and archives under ``archive/``) are rendered from it so the memory stays
**auditable and editable by humans** — a deliberate design property: the compact
memory is not a black box, it is a file you can read and correct.

Entry kinds:

* ``fact``        — a durable piece of knowledge about the user / world.
* ``conclusion``  — a decision or settled position ("concluded X, confirmed").
* ``preference``  — a stable user value, goal, or style preference.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

ENTRY_KINDS = ("fact", "conclusion", "preference")

_SAFE_ID = re.compile(r"[^a-z0-9_-]+")


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class MemoryEntry:
    """A single unit of compact memory."""

    text: str
    kind: str = "fact"
    source_turn: int = 0
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)
    uses: int = 0
    last_used_at: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = _now()
        if not self.updated_at:
            self.updated_at = _now()
        if self.kind not in ENTRY_KINDS:
            raise ValueError(f"kind must be one of {ENTRY_KINDS}")
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("MemoryEntry text must be non-empty")

    def touch(self) -> None:
        self.updated_at = _now()

    def mark_used(self) -> None:
        """Record that the model referenced this entry in an answer."""
        self.uses += 1
        self.last_used_at = _now()

    def token_overlap(self, other: "MemoryEntry") -> float:
        """Token-set Jaccard similarity; used for rule-based dedupe/merge."""
        a = set(_tokenize(self.text))
        b = set(_tokenize(other.text))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _kind_icon(kind: str) -> str:
    return {"fact": "•", "conclusion": "◆", "preference": "★"}.get(kind, "•")


class MemoryStore:
    """Loads/saves memory entries to a state directory and renders Markdown views."""

    def __init__(self, state_dir: str | Path = ".agent-memory") -> None:
        self.root = Path(state_dir)
        self.archive_dir = self.root / "archive"
        self.state_path = self.root / "state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[MemoryEntry] = []
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = [MemoryEntry.from_dict(d) for d in data.get("entries", [])]
        else:
            self._entries = []

    def save(self) -> None:
        """Atomic write (tmp file + rename) to avoid corrupting memory on crash."""
        payload = {"version": 2, "entries": [e.to_dict() for e in self._entries]}
        tmp = self.state_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.state_path)

    # -- query --------------------------------------------------------------

    def all(self, kind: Optional[str] = None) -> List[MemoryEntry]:
        if kind is None:
            return list(self._entries)
        return [e for e in self._entries if e.kind == kind]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    # -- mutation -----------------------------------------------------------

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        self._entries.append(entry)
        return entry

    def remove(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        return len(self._entries) < before

    def replace(self, entry: MemoryEntry) -> None:
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = entry
                return
        self._entries.append(entry)

    def size_bytes(self) -> int:
        return len(json.dumps([e.to_dict() for e in self._entries]))

    # -- rendering ----------------------------------------------------------

    def render_markdown(self, kinds: Optional[List[str]] = None) -> str:
        """Render active entries as human-readable Markdown.

        ``kinds=[\"fact\"]`` renders the memory file; ``kinds=[\"conclusion\",\"preference\"]``
        renders the perspectives file.
        """
        entries = self._entries
        if kinds is not None:
            entries = [e for e in entries if e.kind in kinds]

        lines: List[str] = []
        for kind in ("fact", "conclusion", "preference"):
            group = [e for e in entries if e.kind == kind]
            if not group:
                continue
            heading = {"fact": "## Memory", "conclusion": "## Conclusions", "preference": "## Preferences"}[kind]
            lines.append(heading)
            lines.append("")
            for e in sorted(group, key=lambda x: x.updated_at):
                lines.append(f"- {_kind_icon(kind)} {e.text}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_memory_md(self) -> Path:
        path = self.root / "memory.md"
        path.write_text(self.render_markdown(kinds=["fact"]), encoding="utf-8")
        return path

    def write_perspectives_md(self) -> Path:
        path = self.root / "perspectives.md"
        path.write_text(self.render_markdown(kinds=["conclusion", "preference"]), encoding="utf-8")
        return path

    def write_all_md(self) -> None:
        self.write_memory_md()
        self.write_perspectives_md()

    # -- archiving ----------------------------------------------------------

    def archive_entries(self, entry_ids: List[str], era: Optional[str] = None) -> int:
        """Move entries out of the active store into an era-dated archive file.

        Returns the number of entries moved. Archived entries keep their full text
        (audit trail) but stop consuming active-context tokens.
        """
        if era is None:
            era = datetime.date.today().isoformat()
        safe_era = _SAFE_ID.sub("-", era.lower()) or "archive"
        archive_path = self.archive_dir / f"{safe_era}.md"
        archived: List[MemoryEntry] = []
        for eid in entry_ids:
            entry = self.get(eid)
            if entry is not None:
                archived.append(entry)
                self.remove(eid)

        if archived:
            with open(archive_path, "a", encoding="utf-8") as fh:
                fh.write(f"<!-- archived {_now()} -->\n")
                for e in archived:
                    fh.write(f"- {_kind_icon(e.kind)} [{e.kind}] {e.text}\n")
                fh.write("\n")
        return len(archived)
