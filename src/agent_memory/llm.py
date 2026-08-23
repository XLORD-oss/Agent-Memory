"""Model client layer.

``Client`` is a tiny protocol so the whole library works against any backend.
``OpenAICompatClient`` talks to any OpenAI-compatible chat-completions endpoint
(OpenAI, together, vLLM, llama.cpp, ...). ``MockModel`` is a deterministic
emulator of the *documented* LLM failure modes (context rot, lost-in-the-middle,
sycophantic flipping) so the benchmark harnesses run fully offline in CI while
real models plug in with zero code changes.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

# A message is {"role": ..., "content": ...}
Message = Dict[str, str]


class Client(Protocol):
    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        ...


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "openai package not installed; run `pip install 'agent-memory[llm]'`"
            ) from exc
        self.model = model
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **client_kwargs,
        )

    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        temperature = kwargs.pop("temperature", 0)
        model = kwargs.pop("model", self.model)
        resp = self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Mock model — deterministic emulation of documented failure modes
# ---------------------------------------------------------------------------

FACT_MARKER = re.compile(r"\[\[FACT:(\d+)\]\]\s*=\s*([^\n\]]+)")


class MockModel:
    """Scriptable model that emulates documented LLM failure modes.

    ``mode="context_rot"`` emulates lost-in-the-middle + length rot:

    * The model "retrieves" a fact only if it is near the start or end of the
      context (U-shaped attention) and the total context is short enough.
    * It replies ``<VALUE>`` when it retrieves the fact, else ``UNKNOWN``.

    ``mode="sycophancy"`` emulates self-anchoring + social pressure:

    * If the prompt includes the model's own prior answer being contradicted by
      the user (full-history condition), it flips to the user's stated value,
      with flip probability growing with the number of push-backs.
    * If the context contains no prior assistant turns (compact-memory
      condition), it answers correctly and does not flip.

    The real-model path replaces ``MockModel`` with ``OpenAICompatClient`` and
    the same prompt templates; scoring stays identical.
    """

    def __init__(self, mode: str = "context_rot", seed: int = 0) -> None:
        self.mode = mode
        self.rng = random.Random(seed)
        self.last_stats: Dict[str, Any] = {}

    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        if self.mode == "context_rot":
            return self._context_rot(messages)
        if self.mode == "sycophancy":
            return self._sycophancy(messages)
        raise ValueError(f"unknown mock mode {self.mode!r}")

    # -- context rot --------------------------------------------------------

    def _context_rot(self, messages: List[Message]) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        question = _extract_question(prompt)
        if question is None:
            return "UNKNOWN"

        facts = list(FACT_MARKER.finditer(prompt))
        matches = [f for f in facts if f.group(1) == str(question["fact_id"])]
        if not matches:
            return "UNKNOWN"

        # Position of the fact's first occurrence among all fact markers.
        pos = facts.index(matches[0])
        n_facts = max(1, len(facts))
        relative = pos / n_facts if n_facts > 1 else 0.5
        length_tokens = len(prompt) / 4.0

        # Lost-in-the-middle is a *long-context* phenomenon: in a short context
        # (e.g. a compact memory file) every position is reachable, so the
        # positional penalty only applies once the context is long enough.
        short = length_tokens < 1500
        in_middle = (not short) and 0.15 < relative < 0.85
        # Emulate length rot: reliability decays as total context grows.
        rot_penalty = 0.0 if short else min(1.0, length_tokens / 40_000.0)

        found = (not in_middle) and (self.rng.random() > rot_penalty * 0.6)
        self.last_stats = {
            "relative_pos": round(relative, 3),
            "length_tokens": int(length_tokens),
            "in_middle": in_middle,
            "found": found,
        }
        return matches[0].group(2).strip() if found else "UNKNOWN"

    # -- sycophancy ---------------------------------------------------------

    def _sycophancy(self, messages: List[Message]) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        q = _extract_question(prompt)
        if q is None:
            return ""

        has_prior_assistant = any(m.get("role") == "assistant" for m in messages)
        # Count explicit user push-backs ("the answer is <wrong>").
        pushbacks = len(re.findall(r"\[\[PUSH:([^\]]+)\]\]", prompt))

        if not has_prior_assistant:
            # Compact-memory condition: nothing to anchor onto -> answer correctly.
            self.last_stats = {"flipped": False, "reason": "no prior assistant output"}
            return q["correct"]

        # Full-history condition: model sees its own prior answer being
        # contradicted. Flip probability grows with social pressure.
        flip_prob = 0.35 + 0.18 * pushbacks
        flipped = self.rng.random() < min(flip_prob, 0.95)
        self.last_stats = {
            "flipped": flipped,
            "pushbacks": pushbacks,
            "flip_prob": round(flip_prob, 2),
        }
        return q["wrong"] if flipped else q["correct"]


# ---------------------------------------------------------------------------
# Prompt helpers used by both mock and real paths
# ---------------------------------------------------------------------------

def _extract_question(prompt: str) -> Optional[Dict[str, str]]:
    """Parse a ``QUESTION <id> ...`` block carrying correct/wrong answers.

    Real models see the same template; the mock reads the annotations directly.
    """
    m = re.search(r"\[\[QUESTION:(\d+)\]\]", prompt)
    if not m:
        return None
    qid = m.group(1)
    correct = re.search(r"\[\[CORRECT:([^\]]+)\]\]", prompt)
    wrong = re.search(r"\[\[WRONG:([^\]]+)\]\]", prompt)
    if not correct or not wrong:
        return None
    return {
        "fact_id": qid,
        "correct": correct.group(1).strip(),
        "wrong": wrong.group(1).strip(),
    }


def find_fact_values(prompt: str) -> List[Tuple[str, str]]:
    """All ``[[FACT:id]] = value`` pairs present in a prompt."""
    return [(m.group(1), m.group(2).strip()) for m in FACT_MARKER.finditer(prompt)]
