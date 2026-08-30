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
from .policy import GENERAL, TASK_PROFILES, MemoryPolicy, detect_profile, score_item
from .storage import PIN_PRIORITY, PROTECTED_KINDS, MemoryEntry, MemoryStore
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
        archive_policy: str = "usage",
        usage_threshold: float = 0.4,
        policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.store = store or MemoryStore(state_dir)
        self.distiller = distiller or RuleDistiller()
        self.memory_cap_tokens = memory_cap_tokens
        self.recent_window = recent_window
        # "usage" archives least-valued entries first (scored by the active
        # policy); "oldest" archives by age (previous behavior).
        self.archive_policy = archive_policy
        # Fraction of an entry's tokens that must appear in an answer for the
        # entry to count as "used". Zero API cost: pure token overlap.
        self.usage_threshold = usage_threshold
        # The unified scoring policy: drives BOTH context selection and
        # retention. Defaults to the fresh-chat "general" profile.
        self.policy: MemoryPolicy = policy if policy is not None else GENERAL

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

    # -- the map: structured entry constructors ----------------------------

    def add_entry(
        self,
        text: str,
        kind: str = "fact",
        tags: Optional[List[str]] = None,
        priority: float = 1.0,
        links: Optional[List[str]] = None,
        stance: str = "",
        domain: str = "",
        pinned: bool = False,
    ) -> MemoryEntry:
        """Directly add a structured memory entry (bypasses distillation).

        The low-level constructor for the memory-as-map: facts, conclusions,
        preferences, principles, arguments, perspectives, profile entries — with
        edges (``links``) to other entries.
        """
        entry = MemoryEntry(
            text=text,
            kind=kind,
            tags=tags or [],
            priority=priority,
            links=links or [],
            stance=stance,
            domain=domain,
            pinned=pinned,
        )
        self.store.add(entry)
        self.store.save()
        self.store.write_all_md()
        return entry

    def add_principle(self, text: str, domain: str = "", priority: float = 1.5) -> MemoryEntry:
        """Bake in a first principle / axiom — foundational, never evicted."""
        return self.add_entry(text, kind="principle", domain=domain, priority=priority)

    def add_profile_entry(self, field: str, text: str) -> MemoryEntry:
        """Add to the user profile. ``field`` ∈ identity|domain|style|constraint|goal."""
        field = field.lower()
        if field not in ("identity", "domain", "style", "constraint", "goal"):
            raise ValueError("profile field must be identity|domain|style|constraint|goal")
        return self.add_entry(text, kind="profile", tags=[field], priority=1.4)

    def add_argument(self, claim: str, premises: Optional[List[str]] = None,
                     domain: str = "") -> MemoryEntry:
        """Record a logical argument: a claim deriving from linked premise entries."""
        return self.add_entry(
            text=claim, kind="argument", links=premises or [], domain=domain, priority=1.1
        )

    def add_perspective(self, question: str, stance: str, reasoning: str,
                        links: Optional[List[str]] = None) -> MemoryEntry:
        """Record a viewpoint on an open question (stance: for/against/open).

        Multiple perspectives on the same question can coexist — memory as a
        map holds conflicting views, not a single forced conclusion.
        """
        entry = self.add_entry(
            text=reasoning, kind="perspective", tags=[question],
            stance=stance, links=links or [],
        )
        return entry

    def link(self, entry_id: str, *other_ids: str) -> None:
        """Draw edges in the memory graph: entry_id → other entries."""
        entry = self.store.get(entry_id)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        entry.link(*other_ids)
        self.store.save()

    def related(self, entry_id: str) -> List[MemoryEntry]:
        """Traverse the map: entries linked from ``entry_id`` (resolved, live)."""
        entry = self.store.get(entry_id)
        if entry is None:
            return []
        out = []
        for oid in entry.links:
            other = self.store.get(oid)
            if other is not None:
                out.append(other)
        return out

    def process_turn(
        self,
        user_text: str,
        assistant_text: Optional[str] = None,
        archive: bool = True,
        track_usage: bool = True,
    ) -> int:
        """End-to-end: ingest, distill, merge, optionally track usage and archive.

        Returns the number of new entries committed.
        """
        self.ingest(user_text, assistant_text)
        ext = self.distill_pending()
        new = self.merge(ext)
        if track_usage and assistant_text:
            self._track_usage(assistant_text)
        if archive:
            self.archive_if_needed()
        return new

    # -- usage tracking (learned cache) -------------------------------------

    def _track_usage(self, reply: str) -> int:
        """Count how many memory entries the model's answer actually referenced.

        Zero-API-cost heuristic: an entry counts as used when a threshold
        fraction of its content tokens appears in the reply. This turns memory
        into a **learned cache** — entries your sessions keep referencing are
        protected from eviction; entries that never get referenced are archived
        first, regardless of age.

        Returns the number of entries marked used.
        """
        from .storage import _tokenize

        reply_tokens = set(_tokenize(reply))
        if not reply_tokens:
            return 0
        used_any = False
        for entry in self.store.all():
            entry_tokens = set(_tokenize(entry.text))
            if not entry_tokens:
                continue
            hit = len(entry_tokens & reply_tokens) / len(entry_tokens)
            if hit >= self.usage_threshold:
                entry.mark_used()
                used_any = True
        if used_any:
            self.store.save()
        return sum(1 for e in self.store.all() if e.uses and e.last_used_at)

    def usage_report(self, top: Optional[int] = None) -> List[dict]:
        """Entries sorted by times referenced — the memory's real footprint."""
        entries = sorted(
            self.store.all(), key=lambda e: e.uses, reverse=True
        )
        if top:
            entries = entries[:top]
        return [
            {"text": e.text, "kind": e.kind, "uses": e.uses, "last_used_at": e.last_used_at}
            for e in entries
        ]

    # -- context ------------------------------------------------------------

    def recent_user_turns(self, n: Optional[int] = None) -> List[dict]:
        n = n if n is not None else self.recent_window
        return list(self._raw_turns[-n:])

    def build_context(
        self,
        user_text: str = "",
        mode: Optional[str] = None,
        profile: str = "general",
        policy: Optional["MemoryPolicy"] = None,
        task_window: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> Context:
        """Assemble a prompt under the unified memory policy.

        ``profile`` picks a preset policy ("general" | "coding" | "research" |
        "writing" | "auto"). ``policy`` overrides with a fully custom
        ``MemoryPolicy`` (the first-principles customization surface). ``mode``
        is legacy sugar: ``"minimal"`` → general profile, ``"task"`` → the
        coding profile's working memory. ``task_window`` and ``budget`` override
        the policy's defaults.
        """
        if mode is not None:  # legacy sugar
            profile = "coding" if mode == "task" else "general"
        if profile == "auto":
            probe = user_text
            if not probe and self._raw_turns:
                probe = self._raw_turns[-1]["user"] or ""
            profile = detect_profile(probe)
        ctx = Context.build(
            self,
            user_text,
            profile=profile,
            policy=policy,
            task_window=task_window,
            budget=budget,
        )
        self._prompt_tokens_total += ctx.prompt_tokens
        return ctx

    # -- priorities (the customization surface) ------------------------------

    def set_policy(self, policy: "MemoryPolicy") -> None:
        """Swap the scoring policy (drives context selection AND retention)."""
        self.policy = policy

    def set_policy_profile(self, profile: str) -> None:
        """Swap to a named preset: general | coding | research | writing."""
        if profile not in TASK_PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(TASK_PROFILES)}")
        self.policy = TASK_PROFILES[profile]

    def set_priority(self, entry_id: str, priority: float) -> None:
        """Explicit priority for an entry (1.0 neutral, >1 boosted).

        The strongest knob: a priority-10 fact beats a fresh priority-1 fact
        for context budget and eviction, regardless of age.
        """
        entry = self.store.get(entry_id)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        entry.priority = max(0.0, float(priority))
        self.store.save()

    def prioritize(self, text_contains: str, priority: float) -> int:
        """Set priority on every entry whose text contains ``text_contains``.

        Convenience for "remember *this topic* stronger": returns the number of
        entries adjusted.
        """
        n = 0
        for e in self.store.all():
            if text_contains.lower() in e.text.lower():
                e.priority = max(0.0, float(priority))
                n += 1
        if n:
            self.store.save()
        return n

    def pin(self, entry_id: str) -> None:
        """Pin an entry: unbounded priority — always in context, never evicted."""
        entry = self.store.get(entry_id)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        entry.pinned = True
        self.store.save()

    def unpin(self, entry_id: str) -> None:
        entry = self.store.get(entry_id)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        entry.pinned = False
        self.store.save()

    # -- archiving ----------------------------------------------------------

    def archive_if_needed(self, policy: Optional[str] = None) -> int:
        """Evict entries while active memory exceeds the cap.

        ``policy="usage"`` (default) archives the least-VALUED entries first,
        scored by the active ``self.policy`` (priority + usage + recency +
        task affinity). Pinned entries and decisions (conclusions) survive.
        ``policy="oldest"`` archives by age.

        Returns the number of entries archived this call.
        """
        policy = policy or self.archive_policy
        moved = 0
        while self.memory_tokens() > self.memory_cap_tokens:
            entries = list(self.store.all())
            if not entries:
                break
            victim = _pick_victim(entries, engine=self, archive_policy=policy)
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
        return self.store.render_markdown(kinds=["conclusion", "preference", "perspective"])

    @property
    def principles_md(self) -> str:
        return self.store.render_markdown(kinds=["principle"])

    @property
    def profile_md(self) -> str:
        return self.store.render_markdown(kinds=["profile"])

    def memory_tokens(self) -> int:
        return (
            estimate_tokens(self.memory_md)
            + estimate_tokens(self.perspectives_md)
            + estimate_tokens(self.principles_md)
            + estimate_tokens(self.profile_md)
        )

    @property
    def entry_count(self) -> int:
        return len(self.store.all())

    @property
    def raw_turn_count(self) -> int:
        return len(self._raw_turns)

    def total_tokens_processed(self) -> int:
        """Total tokens this engine has placed in model context so far."""
        return self._prompt_tokens_total


def _pick_victim(
    entries: List[MemoryEntry],
    engine: Optional["MemoryEngine"] = None,
    archive_policy: str = "usage",
) -> Optional[MemoryEntry]:
    """Choose the entry to archive first — the retention side of the policy.

    ``usage`` (default): the lowest-SCORED entry under the engine's active
    ``MemoryPolicy``. Pinned entries and the structural foundations
    (conclusions, principles) are excluded from candidacy. ``oldest``: by age.
    """
    candidates = [e for e in entries if e.kind not in PROTECTED_KINDS and not e.pinned]
    if not candidates:
        return None  # never evict the foundations; pinned are untouchable
    if archive_policy == "oldest":
        return min(candidates, key=lambda e: e.updated_at)

    if engine is not None:
        current_turn = engine.raw_turn_count

        def score(e: MemoryEntry) -> float:
            age = max(0, current_turn - e.source_turn) if e.source_turn else 0
            affinity = engine.policy.affinity(e.text, e.tags)
            return score_item(
                e.text, e.kind, e.effective_priority(), e.uses, age,
                engine.policy, affinity=affinity,
            )

        return min(candidates, key=score)

    # Fallback without an engine: lowest priority, then lowest uses, then oldest.
    return min(candidates, key=lambda e: (e.effective_priority(), e.uses, e.updated_at))
