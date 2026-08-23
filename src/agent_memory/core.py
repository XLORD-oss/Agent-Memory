"""The compact memory engine.

``MemoryEngine`` is a drop-in memory layer for agents:

1. **ingest** — each raw turn is logged (for audit) but *not* scheduled for replay.
2. **distill** — new turns are distilled (rules or an LLM) into candidate facts,
   conclusions, and preferences. Only what is genuinely new and durable survives.
3. **merge** — candidates are deduplicated against existing memory; new entries are
   committed. ``memory.md`` and ``perspectives.md`` are re-rendered so the memory
   stays human-readable.
4. **archive** — when the active memory exceeds the token cap, the oldest entries
   are split off into era-dated ``archive/*.md`` files, keeping the active context
   small and bounded.
5. **build_context** — assembles the fresh-chat prompt: memory + perspectives +
   recent *user* turns only. Prior assistant raw outputs are never replayed.

The result: total tokens processed grow ~linearly with conversation length instead
of quadratically, and the model never has to search a rotting transcript.
"""

from __future__ import annotations

from typing import List, Optional

from .context import Context
from .distiller import DistillerLike, Extraction, RuleDistiller
from .llm import Client
from .storage import MemoryEntry, MemoryStore
from .tokens import estimate_tokens

DEFAULT_MEMORY_CAP_TOKENS = 3000
DEFAULT_RECENT_WINDOW = 4
# Token-overlap Jaccard at or above which a candidate is treated as a duplicate.
# 0.85 means "identical except trivial wording" — high enough that facts differing
# in one informative token (e.g. "us-east-1" vs "us-east-2") are NOT collapsed.
MERGE_THRESHOLD = 0.85


class MemoryEngine:
    """Self-maintaining compact memory for a long-running conversation."""

    def __init__(
        self,
        state_dir: str = ".agent-memory",
        store: Optional[MemoryStore] = None,
        distiller: Optional[DistillerLike] = None,
        memory_cap_tokens: int = DEFAULT_MEMORY_CAP_TOKENS,
        recent_window: int = DEFAULT_RECENT_WINDOW,
    ) -> None:
        self.store = store or MemoryStore(state_dir)
        self.distiller = distiller or RuleDistiller()
        self.memory_cap_tokens = memory_cap_tokens
        self.recent_window = recent_window

        # Audit log of raw turns (kept for accountability, never replayed).
        self._raw_turns: List[dict] = []
        self._pending_start = 0
        self._turn_number = max((e.source_turn for e in self.store.all()), default=0)
        # Total tokens this engine has placed in model context (for cost validation).
        self._prompt_tokens_total = 0

    # -- ingest / distill / merge -------------------------------------------

    def ingest(self, user_text: str, assistant_text: Optional[str] = None) -> int:
        self._turn_number += 1
        self._raw_turns.append(
            {"turn": self._turn_number, "user": user_text, "assistant": assistant_text}
        )
        return self._turn_number

    def distill_pending(self) -> Extraction:
        """Distill raw turns that have not yet been distilled."""
        ext = Extraction()
        for t in self._raw_turns[self._pending_start:]:
            turn_ext = self.distiller.distill(
                t["user"], t.get("assistant") or "", t["turn"]
            )
            ext.facts.extend(turn_ext.facts)
            ext.conclusions.extend(turn_ext.conclusions)
            ext.preferences.extend(turn_ext.preferences)
        self._pending_start = len(self._raw_turns)
        return ext

    def merge(self, ext: Extraction, threshold: float = MERGE_THRESHOLD) -> int:
        """Commit distilled entries, deduplicating against existing memory.

        Returns the number of *new* entries committed.
        """
        committed = 0
        for cand in ext.all():
            if self._find_duplicate(cand, threshold) is not None:
                continue
            self.store.add(cand)
            committed += 1
        self.store.save()
        self.store.write_all_md()
        return committed

    def _find_duplicate(self, cand: MemoryEntry, threshold: float) -> Optional[MemoryEntry]:
        for existing in self.store.all(kind=cand.kind):
            if existing.token_overlap(cand) >= threshold:
                # Keep the more informative wording; refresh timestamp.
                if len(cand.text) > len(existing.text):
                    existing.text = cand.text
                existing.touch()
                return existing
        return None

    def process_turn(
        self, user_text: str, assistant_text: Optional[str] = None, archive: bool = True
    ) -> int:
        """End-to-end: ingest, distill, merge, and optionally archive. Returns new entries."""
        self.ingest(user_text, assistant_text)
        ext = self.distill_pending()
        new = self.merge(ext)
        if archive:
            self.archive_if_needed()
        return new

    # -- context ------------------------------------------------------------

    def recent_user_turns(self, n: Optional[int] = None) -> List[dict]:
        n = n if n is not None else self.recent_window
        return list(self._raw_turns[-n:])

    def build_context(self, user_text: str = "") -> Context:
        ctx = Context.build(self, user_text)
        self._prompt_tokens_total += ctx.prompt_tokens
        return ctx

    # -- archiving ----------------------------------------------------------

    def archive_if_needed(self) -> int:
        """Move oldest entries to the archive while active memory exceeds the cap.

        Returns the number of entries archived this call.
        """
        moved = 0
        while self.memory_tokens() > self.memory_cap_tokens:
            oldest = sorted(self.store.all(), key=lambda e: e.updated_at)
            if not oldest:
                break
            # Keep conclusions (decisions) and the very latest entry in context;
            # archive the oldest non-conclusion / oldest overall first.
            victim = _pick_victim(oldest)
            if victim is None:
                break
            n = self.store.archive_entries([victim.id])
            if n == 0:
                break
            moved += n
        if moved:
            self.store.save()
            self.store.write_all_md()
        return moved

    # -- read-only views ----------------------------------------------------

    @property
    def memory_md(self) -> str:
        return self.store.render_markdown(kinds=["fact"])

    @property
    def perspectives_md(self) -> str:
        return self.store.render_markdown(kinds=["conclusion", "preference"])

    def memory_tokens(self) -> int:
        return estimate_tokens(self.memory_md) + estimate_tokens(self.perspectives_md)

    @property
    def entry_count(self) -> int:
        return len(self.store.all())

    @property
    def raw_turn_count(self) -> int:
        return len(self._raw_turns)

    def total_tokens_processed(self) -> int:
        """Total tokens this engine has placed in model context so far."""
        return self._prompt_tokens_total


def _pick_victim(entries: List[MemoryEntry]) -> Optional[MemoryEntry]:
    """Choose what to archive first: oldest non-conclusion, else oldest."""
    non_conclusions = [e for e in entries if e.kind != "conclusion"]
    if non_conclusions:
        return non_conclusions[0]
    return entries[0] if entries else None
