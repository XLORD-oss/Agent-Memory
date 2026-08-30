"""Agent-Memory — compact, self-maintaining memory for LLM agents.

Replace full-transcript replay with a distilled fact store that only grows when a
genuinely new fact appears: token cost goes from quadratic to near-linear, the
model never has to search a rotting transcript (context rot), and prior assistant
outputs are kept out of context so the model cannot anchor on its own past words.

Key modules:

* ``core.MemoryEngine`` — ingest → distill → merge → archive → build_context.
* ``storage``           — file-backed, human-readable ``memory.md`` / ``perspectives.md``.
* ``distiller``         — rules-based and LLM-based extraction.
* ``tokens``            — the O(T²) vs O(T) cost model.
* ``llm``               — OpenAI-compatible client + offline mock of failure modes.
"""

from .core import MemoryEngine
from .distiller import Extraction, LLMDistiller, RuleDistiller
from .llm import MockModel, OpenAICompatClient
from .policy import GENERAL, CODING, RESEARCH, WRITING, MemoryPolicy, TASK_PROFILES, detect_profile
from .storage import MemoryEntry, MemoryStore
from .tokens import (
    compact_total_tokens,
    compare,
    cost_table,
    estimate_tokens,
    full_history_total_tokens,
)

__all__ = [
    "MemoryEngine",
    "MemoryEntry",
    "MemoryStore",
    "Extraction",
    "RuleDistiller",
    "LLMDistiller",
    "MockModel",
    "OpenAICompatClient",
    "MemoryPolicy",
    "TASK_PROFILES",
    "GENERAL",
    "CODING",
    "RESEARCH",
    "WRITING",
    "detect_profile",
    "estimate_tokens",
    "full_history_total_tokens",
    "compact_total_tokens",
    "compare",
    "cost_table",
]

__version__ = "0.1.0"
