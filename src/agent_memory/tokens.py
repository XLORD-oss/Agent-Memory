"""Token estimation and the cost model that quantifies the framework's core claim.

The headline arithmetic, which holds regardless of model:

* **Full-history replay** — the naive default where every turn re-sends the whole
  raw transcript. Turn N carries ~N turns of tokens, so total tokens processed over
  a T-turn conversation scale as ``per_turn * T * (T + 1) / 2`` → **O(T²)**.

* **Compact memory** — the active context is bounded (a curated memory file plus a
  small rolling window of recent turns). Memory only grows when a *genuinely new*
  fact appears, and older entries are split off to an archive, so total tokens
  processed grow **~linearly in T**.

This module is pure Python with no hard dependencies. If ``tiktoken`` is installed
it is used for accurate estimation; otherwise a ``len/4`` heuristic (average English
token is ~4 characters) is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore

    _HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAVE_TIKTOKEN = False

CHARS_PER_TOKEN = 4.0
_enc_cache: dict = {}


def _encoder(name: str = "cl100k_base"):
    if not _HAVE_TIKTOKEN:  # pragma: no cover
        return None
    if name not in _enc_cache:
        _enc_cache[name] = tiktoken.get_encoding(name)
    return _enc_cache[name]


def estimate_tokens(text: str, method: str = "auto") -> int:
    """Estimate the token count of ``text``.

    ``method="auto"`` uses tiktoken when available, otherwise a characters/4
    heuristic. ``method="chars4"`` and ``method="words"`` force a heuristic.
    """
    if method == "auto" and _HAVE_TIKTOKEN:  # pragma: no cover - env dependent
        enc = _encoder()
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover
            pass
    if method in ("auto", "chars4"):
        return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
    if method == "words":
        return max(1, math.ceil(len(text.split()) * 1.333))
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def full_history_total_tokens(turns: int, per_turn: int = 500) -> int:
    """Total tokens processed over a T-turn conversation under full-history replay.

    Turn N re-sends turns 1..N-1, so the total is per_turn * (1 + 2 + ... + T).
    """
    return per_turn * turns * (turns + 1) // 2


def compact_total_tokens(
    turns: int,
    per_turn: int = 500,
    memory_cap: int = 3000,
    recent_window: int = 4,
    new_fact_tokens: Optional[float] = None,
) -> int:
    """Total tokens processed under compact memory.

    Memory grows by ``new_fact_tokens`` per turn (only genuinely new facts commit;
    a fact is a heavily compressed version of its source turn) until it hits
    ``memory_cap``, after which older entries are archived and the active context
    stays flat. A small rolling window of ``recent_window`` raw turns keeps
    conversational coherence.
    """
    if new_fact_tokens is None:
        new_fact_tokens = per_turn / 8.0  # default ~8x compression of turns -> facts
    total = 0
    for n in range(1, turns + 1):
        memory = min(n * new_fact_tokens, memory_cap)
        total += int(memory + recent_window * per_turn)
    return total


@dataclass
class CostComparison:
    turns: int
    per_turn: int
    memory_cap: int
    recent_window: int
    new_fact_tokens: float
    full_history: int
    compact: int

    @property
    def ratio(self) -> float:
        """compact / full_history — how much of the naive bill remains."""
        return self.compact / self.full_history if self.full_history else float("inf")

    @property
    def saved_fraction(self) -> float:
        return 1.0 - self.ratio

    def table_row(self) -> List[str]:
        return [
            f"{self.turns:,}",
            f"{self.full_history:,}",
            f"{self.compact:,}",
            f"{self.ratio:.3f}",
            f"{self.saved_fraction * 100:.1f}%",
        ]


def compare(
    turns: int = 500,
    per_turn: int = 500,
    memory_cap: int = 3000,
    recent_window: int = 4,
    new_fact_tokens: Optional[float] = None,
) -> CostComparison:
    return CostComparison(
        turns=turns,
        per_turn=per_turn,
        memory_cap=memory_cap,
        recent_window=recent_window,
        new_fact_tokens=new_fact_tokens if new_fact_tokens is not None else per_turn / 8.0,
        full_history=full_history_total_tokens(turns, per_turn),
        compact=compact_total_tokens(turns, per_turn, memory_cap, recent_window, new_fact_tokens),
    )


def cost_table(
    turn_counts=(50, 100, 200, 500, 1000, 2000, 5000),
    per_turn: int = 500,
    memory_cap: int = 3000,
    recent_window: int = 4,
) -> str:
    """Render a Markdown table comparing the two policies across conversation lengths."""
    rows = [compare(t, per_turn, memory_cap, recent_window) for t in turn_counts]
    header = ["Turns", "Full history (tokens)", "Compact memory (tokens)", "Ratio", "Saved"]
    sep = "|---|---|---|---|---|"

    def _row(cells) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    return "\n".join([_row(header), sep] + [_row(r.table_row()) for r in rows])


def plot_costs(
    turn_counts=(50, 100, 200, 500, 1000, 2000),
    per_turn: int = 500,
    memory_cap: int = 3000,
    recent_window: int = 4,
    out_path: str = "token_cost.png",
) -> None:
    """Plot total tokens processed vs. turns for both policies (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise RuntimeError("matplotlib not installed; pip install 'agent-memory[plot]'")

    xs = list(turn_counts)
    full = [full_history_total_tokens(t, per_turn) for t in xs]
    compact = [compact_total_tokens(t, per_turn, memory_cap, recent_window) for t in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [f / 1e6 for f in full], "o-", label=f"Full-history replay (O(T²))")
    ax.plot(xs, [c / 1e6 for c in compact], "s-", label=f"Compact memory (O(T))")
    ax.set_xlabel("Conversation turns (T)")
    ax.set_ylabel("Total tokens processed (millions)")
    ax.set_title("Total tokens processed over a T-turn conversation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
