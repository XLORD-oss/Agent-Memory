"""Prompt construction for the compact-memory context.

The key design property is the **fresh-chat contract**: prior *assistant* raw
outputs are never placed back in context. The model sees only the distilled
memory, the distilled perspectives, and a small rolling window of recent *user*
turns for conversational flow. This is the mechanism the framework claims reduces
self-anchoring and sycophantic drift — the model cannot treat "I already said X"
as evidence, because its raw past words are not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

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

    ``mode`` selects the presentation strategy:

    * ``minimal`` — the fresh-chat contract: memory + perspectives + recent
      USER turns only. Prior assistant outputs are never replayed. Default.
    * ``task`` — adds a bounded verbatim ``WORKING MEMORY`` section containing
      the last ``task_window`` raw turns (user AND assistant), explicitly
      delimited. This is the one deliberate exception to the fresh-chat rule,
      for tasks that need working-memory precision (code, derivations, edits).
      The window is bounded so it cannot rot into a full replay.
    """

    system: str = SYSTEM_PROMPT
    mode: str = "minimal"
    memory: str = ""
    perspectives: str = ""
    recent_user_turns: List[str] = field(default_factory=list)
    work_window: List[dict] = field(default_factory=list)
    user_text: str = ""
    prompt_tokens: int = 0

    @property
    def user_prompt(self) -> str:
        sections: List[str] = []
        if self.memory.strip():
            sections.append("MEMORY\n" + self.memory.strip())
        if self.perspectives.strip():
            sections.append("PERSPECTIVES\n" + self.perspectives.strip())
        if self.mode == "task" and self.work_window:
            lines = []
            for t in self.work_window:
                if t.get("user"):
                    lines.append(f"- user: {t['user']}")
                if t.get("assistant"):
                    lines.append(f"- assistant: {t['assistant']}")
            sections.append(
                "WORKING MEMORY (the active task, verbatim and bounded — the exception "
                "to the fresh-chat rule, scoped to this task)\n" + "\n".join(lines)
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

    @classmethod
    def build(
        cls,
        engine: "MemoryEngine",
        user_text: str = "",
        mode: str = "minimal",
        task_window: int = 8,
    ) -> "Context":
        work_window: List[dict] = []
        recent: List[str] = []
        if mode == "task":
            work_window = list(engine._raw_turns[-task_window:])
        else:
            recent = [t["user"] for t in engine.recent_user_turns()]
        ctx = cls(
            mode=mode,
            memory=engine.memory_md,
            perspectives=engine.perspectives_md,
            recent_user_turns=recent,
            work_window=work_window,
            user_text=user_text,
        )
        ctx.prompt_tokens = estimate_tokens(ctx.system) + estimate_tokens(ctx.user_prompt)
        return ctx
