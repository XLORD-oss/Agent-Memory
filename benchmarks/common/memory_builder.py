"""Build the compact-memory condition for the benchmarks.

The benchmarks isolate the *presentation* effect, so the memory condition is
built by a benchmark-controlled distiller (``MarkerDistiller``) that extracts
the ground-truth fact lines verbatim — exactly like a real distillation step
would preserve them, but deterministic so the experiment is reproducible.

Real-model runs can swap in ``LLMDistiller`` (see ``--llm-distill``) to exercise
the production distillation path; the mock runs use ``MarkerDistiller`` so the
harness works offline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import sys

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_memory.core import MemoryEngine  # noqa: E402
from agent_memory.distiller import DistillerLike, Extraction  # noqa: E402
from agent_memory.storage import MemoryEntry  # noqa: E402

FACT_LINE = re.compile(r"\[\[FACT:\d+\]\]\s*=\s*[^\n\]]+")


class MarkerDistiller:
    """Extracts ground-truth fact lines (``[[FACT:id]] = value``) verbatim.

    Conservative: only the marker lines become memory, so memory contains exactly
    the facts the experiment planted — nothing else.
    """

    def distill(self, user_text: str, assistant_text: str = "", turn_number: int = 0) -> Extraction:
        ext = Extraction()
        for line in user_text.splitlines():
            m = FACT_LINE.search(line)
            if m:
                ext.facts.append(
                    MemoryEntry(text=m.group(0).strip(), kind="fact", source_turn=turn_number)
                )
        return ext


class ConclusionDistiller:
    """Records the model's conclusion from a (question, answer) turn.

    Used by the sycophancy benchmark to populate the perspectives file the way
    a production system would: "concluded X" — the decision without the social
    momentum that surrounded it.
    """

    def distill(self, user_text: str, assistant_text: str = "", turn_number: int = 0) -> Extraction:
        ext = Extraction()
        if assistant_text and assistant_text.strip():
            ext.conclusions.append(
                MemoryEntry(
                    text=f"Concluded: for \"{user_text.strip()[:120]}\", the answer is {assistant_text.strip()}.",
                    kind="conclusion",
                    source_turn=turn_number,
                )
            )
        return ext


def build_memory_engine(
    state_dir: str,
    distiller: DistillerLike,
    memory_cap_tokens: int = 3000,
    recent_window: int = 4,
) -> MemoryEngine:
    return MemoryEngine(
        state_dir=state_dir,
        distiller=distiller,
        memory_cap_tokens=memory_cap_tokens,
        recent_window=recent_window,
    )
