"""Prompt construction for the compact-memory context.

Assembly is a **token-budget fill**, not a mode switch: every candidate (a
distilled fact, conclusion, preference, or a recent raw turn) is scored by the
active ``MemoryPolicy`` (see ``policy.py``) and the budget is filled
highest-value-first.

Two regimes fall out of the same vector:

* **Fresh-chat contract (default, ``general`` profile).** Raw assistant turns
  have ``kind_weights["assistant_turn"] = 0.0``, so they are never selected —
  the model cannot anchor on "I already said X" because its raw past words are
  not there. Recent raw USER turns remain eligible for conversational flow.
* **Task memory (e.g. ``coding`` profile).** Recent raw turns (user + assistant)
  compete for the budget with a positive weight, giving bounded working memory
  for code, derivations, and edits. This is the "exception" when the task
  demands it — expressed as a weight, not hardcoded.

``mode="minimal"`` / ``mode="task"`` remain as sugar mapping to the profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from .policy import TASK_PROFILES, MemoryPolicy, score_item, select_by_score
from .tokens import estimate_tokens

if TYPE_CHECKING:  # pragma: no cover
    from .core import MemoryEngine

SYSTEM_PROMPT = """You are an assistant with a compact long-term memory. The MEMORY and \
PERSPECTIVES blocks below are the authoritative record of what is known and what has been \
decided; treat them as facts.

Ground rules:
- Answer from the memory and the current question. You are NOT shown your prior raw replies \
by design, so each turn is judged on its own merits. Do not treat "what I said before" as \
evidence — you cannot see it.
- The memory is maintained by a separate process and may have been corrected. If the user \
contradicts the memory, update your view; do not cling to an earlier position.
- Be direct and truthful. If the memory does not contain what you need, say so rather than \
guessing or agreeing to please the user."""


@dataclass
class Context:
    """A fully assembled compact-memory prompt.

    ``mode`` is informational (which preset was used); the actual selection is
    driven by ``policy``. The rendered sections live in ``memory``,
    ``perspectives``, and exactly one of ``recent_user_turns`` (flow context,
    no assistant content) or ``work_window`` (task memory, may include
    assistant content).
    """

    system: str = SYSTEM_PROMPT
    mode: str = "general"
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    memory: str = ""
    perspectives: str = ""
    recent_user_turns: List[str] = field(default_factory=list)
    work_window: List[dict] = field(default_factory=list)
    user_text: str = ""
    budget_tokens: int = 0
    used_tokens: int = 0
    scored_candidates: int = 0
    prompt_tokens: int = 0

    @property
    def has_working_memory(self) -> bool:
        return any(w.get("assistant") for w in self.work_window) if self.work_window else False

    @property
    def user_prompt(self) -> str:
        sections: List[str] = []
        if self.memory.strip():
            sections.append("MEMORY\n" + self.memory.strip())
        if self.perspectives.strip():
            sections.append("PERSPECTIVES\n" + self.perspectives.strip())
        if self.work_window:
            lines = []
            for t in self.work_window:
                if t.get("user"):
                    lines.append(f"- user: {t['user']}")
                if t.get("assistant"):
                    lines.append(f"- assistant: {t['assistant']}")
            sections.append(
                "WORKING MEMORY (recent turns of the active task, bounded — includes the "
                "model's own recent code/reasoning when the task profile allows)\n"
                + "\n".join(lines)
            )
        elif self.recent_user_turns:
            turns = "\n".join(f"- {t}" for t in self.recent_user_turns)
            sections.append("RECENT CONTEXT (for conversational flow)\n" + turns)
        if self.user_text.strip():
            sections.append("CURRENT USER MESSAGE\n" + self.user_text.strip())
        return "\n\n".join(sections)

    def to_messages(self) -> List[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_prompt},
        ]

    # -- assembly -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        engine: "MemoryEngine",
        user_text: str = "",
        profile: str = "general",
        policy: Optional[MemoryPolicy] = None,
        task_window: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> "Context":
        pol = policy if policy is not None else TASK_PROFILES.get(profile, TASK_PROFILES["general"])
        budget = budget if budget is not None else (pol.budget_tokens or engine.memory_cap_tokens)
        window = task_window if task_window is not None else pol.task_window_turns
        current_turn = engine.raw_turn_count

        # ---- 1. candidate pool: memory entries ----------------------------
        candidates: List[Dict] = []
        for e in engine.store.all():
            age = max(0, current_turn - e.source_turn) if e.source_turn else 0
            score = score_item(
                e.text, e.kind, e.effective_priority(), e.uses, age, pol,
                affinity=pol.affinity(e.text, e.tags),
            )
            candidates.append(
                {"kind": e.kind, "text": e.text, "score": score,
                 "tokens": estimate_tokens(e.text), "entry": e}
            )

        # ---- 2. candidate pool: recent raw turns --------------------------
        # Flow context (last <recent_window> USER turns) is always eligible —
        # conversational coherence under the fresh-chat contract. Task memory
        # (a working-memory window that may include assistant turns) becomes
        # eligible only when the policy's task_window_turns > 0; it is the
        # superset of the flow window so turns are never counted twice.
        flow = engine.recent_user_turns(engine.recent_window)
        if window > 0:
            pool = engine._raw_turns[-max(window, engine.recent_window):]
        else:
            pool = flow
        for t in pool:
            age = max(0, current_turn - t["turn"])
            if t.get("user"):
                candidates.append(
                    {"kind": "user_turn", "text": t["user"], "entry": None,
                     "score": score_item(t["user"], "user_turn", 1.0, 0, age, pol,
                                        pol.affinity(t["user"], [])),
                     "tokens": estimate_tokens(t["user"]),
                     "role": "user", "turn": t["turn"]}
                )
            if t.get("assistant") and window > 0:
                candidates.append(
                    {"kind": "assistant_turn", "text": t["assistant"], "entry": None,
                     "score": score_item(t["assistant"], "assistant_turn", 1.0, 0, age, pol,
                                        pol.affinity(t["assistant"], [])),
                     "tokens": estimate_tokens(t["assistant"]),
                     "role": "assistant", "turn": t["turn"]}
                )

        # ---- 3. fill the budget -------------------------------------------
        ctx = cls(mode=profile, policy=pol, budget_tokens=budget, user_text=user_text)
        ctx.scored_candidates = len(candidates)
        selected = select_by_score(candidates, budget)
        ctx.used_tokens = sum(c["tokens"] for c in selected)

        # ---- 4. render into sections ---------------------------------------
        facts = [c for c in selected if c["kind"] == "fact"]
        pers = [c for c in selected if c["kind"] in ("conclusion", "preference")]
        turns = [c for c in selected if c["kind"] in ("user_turn", "assistant_turn")]

        if facts:
            lines = []
            for c in sorted(facts, key=lambda c: -c["score"]):
                marker = "◆" if c["kind"] == "conclusion" else "★" if c["kind"] == "preference" else "•"
                lines.append(f"- {marker} {c['text']}")
            ctx.memory = "## Memory\n" + "\n".join(lines)

        if pers:
            by_kind = {"## Conclusions": [], "## Preferences": []}
            for c in pers:
                if c["kind"] == "conclusion":
                    by_kind["## Conclusions"].append(c)
                else:
                    by_kind["## Preferences"].append(c)
            chunks = []
            for header in ("## Conclusions", "## Preferences"):
                if by_kind[header]:
                    chunks.append(header + "\n" + "\n".join(f"- ◆ {c['text']}" if "Conclusion" in header else f"- ★ {c['text']}" for c in by_kind[header]))
            ctx.perspectives = "\n\n".join(chunks)

        if turns:
            turns_sorted = sorted(turns, key=lambda c: (c.get("turn", 0), c["kind"]))
            has_assistant = any(c["kind"] == "assistant_turn" for c in turns_sorted)
            if has_assistant:
                grouped: Dict[int, dict] = {}
                for c in turns_sorted:
                    grouped.setdefault(c["turn"], {"user": "", "assistant": ""})
                    if c["kind"] == "user_turn":
                        grouped[c["turn"]]["user"] = c["text"]
                    else:
                        grouped[c["turn"]]["assistant"] = c["text"]
                ctx.work_window = [grouped[t] for t in sorted(grouped)]
            else:
                ctx.recent_user_turns = [c["text"] for c in turns_sorted]

        ctx.prompt_tokens = estimate_tokens(ctx.system) + estimate_tokens(ctx.user_prompt)
        return ctx