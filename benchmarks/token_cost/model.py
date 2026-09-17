"""Token-cost benchmark: the quadratic-vs-linear arithmetic, no model required.

Full-history replay means turn N carries ~N turns of tokens, so total tokens
processed over a T-turn conversation scale as O(T²). Compact memory (capped fact
store + a small rolling window) scales ~O(T). This module prints the comparison
table and, optionally, a plot.

    python -m benchmarks.token_cost.model
    python -m benchmarks.token_cost.model --plot /tmp/token_cost.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent_memory.tokens import (  # noqa: E402
    compact_total_tokens,
    cost_table,
    full_history_total_tokens,
    plot_costs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-turn", type=int, default=500, help="avg raw tokens per turn")
    parser.add_argument("--memory-cap", type=int, default=3000, help="compact memory token cap")
    parser.add_argument("--recent-window", type=int, default=4, help="raw turns kept for flow")
    parser.add_argument("--plot", default=None, help="optional path to save a PNG plot")
    args = parser.parse_args()

    print("Token-cost model: total input tokens processed over a T-turn conversation")
    print(f"(assumptions: ~{args.per_turn} raw tokens/turn, memory capped at "
          f"{args.memory_cap} tokens, {args.recent_window} recent raw turns kept)\n")
    print(cost_table(per_turn=args.per_turn, memory_cap=args.memory_cap, recent_window=args.recent_window))

    t = 1000
    full = full_history_total_tokens(t, args.per_turn)
    compact = compact_total_tokens(t, args.per_turn, args.memory_cap, args.recent_window)
    print(f"\nAt T={t:,} turns:")
    print(f"  full-history replay : {full:>16,} tokens")
    print(f"  compact memory      : {compact:>16,} tokens")
    print(f"  ratio               : {compact / full:.4f}  (a {full / compact:.1f}x reduction)")

    if args.plot:
        plot_costs(per_turn=args.per_turn, memory_cap=args.memory_cap, recent_window=args.recent_window, out_path=args.plot)


if __name__ == "__main__":
    main()
