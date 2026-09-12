"""Estimate the API cost of the real-model benchmark campaign — before spending it.

Run this before asking anyone for credits or compute. It builds the *exact*
prompts the benchmarks would send (no model calls), counts their tokens, and
prices them per model:

    python -m benchmarks.budget                       # default campaign
    python -m benchmarks.budget --turns 200 1000 5000 --seeds 5
    python -m benchmarks.budget --models openai/gpt-4o anthropic/claude-3.5-sonnet
    python -m benchmarks.budget --price openai/gpt-4o=2.50:10.00   # override $/M in:out

Why this exists
---------------
The question "how much compute do you need?" is the first thing a lab, a
professor, or a credit program will ask. The honest answer is small: the
compact-memory condition is a few hundred tokens per call, and the raw
full-transcript condition — the expensive one — is exactly what the framework
is designed to remove. This script makes that number concrete and reproducible.

Prices are **input:output USD per million tokens** and drift over time; the
defaults below are rough mid-2026 OpenRouter list prices. Override with
``--price`` for anything current. Output is *estimated* (the benchmarks ask for
one-value answers; ``--out-tokens`` sets the assumed reply length).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory.context import SYSTEM_PROMPT  # noqa: E402
from agent_memory.tokens import estimate_tokens  # noqa: E402

from benchmarks.common.memory_builder import (  # noqa: E402
    ConclusionDistiller,
    MarkerDistiller,
    build_memory_engine,
)
from benchmarks.context_rot.run import memory_user_prompt, raw_user_prompt  # noqa: E402
from benchmarks.context_rot.tasks import VALUE_POOL, generate_transcript, question_for  # noqa: E402
from benchmarks.sycophancy.tasks import ITEMS, pushback_text  # noqa: E402

# Rough list prices, USD per million tokens (input, output). Override with --price.
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "qwen/qwen-2.5-72b-instruct": (0.35, 0.40),
}


@dataclass
class Line:
    """Token totals for one benchmark cell (one setting, one seed)."""

    benchmark: str
    setting: str
    calls: int
    raw_in: int
    mem_in: int
    out_calls_raw: int
    out_calls_mem: int

    @property
    def total_in(self) -> int:
        return self.raw_in + self.mem_in


class _Plain:
    """A stand-in client that is *not* the mock, so prompts render as real models see them."""


def context_rot_tokens(turns: int, facts: int, seed: int) -> Line:
    transcript = generate_transcript(turns, facts, seed)
    engine = build_memory_engine(
        state_dir=tempfile.mkdtemp(prefix="agent-memory-budget-"),
        distiller=MarkerDistiller(),
    )
    for line in transcript.raw.splitlines():
        engine.process_turn(line)
    mem_ctx = engine.build_context("")

    raw_in = mem_in = 0
    sys_raw = estimate_tokens(SYSTEM_PROMPT)
    sys_mem = estimate_tokens(mem_ctx.system)
    for fact in transcript.facts:
        q = question_for(fact)
        raw_in += sys_raw + estimate_tokens(raw_user_prompt(transcript.raw, q))
        mem_in += sys_mem + estimate_tokens(memory_user_prompt(mem_ctx, q))
    n = len(transcript.facts)
    return Line("context_rot", f"turns={turns} facts={facts}", 2 * n, raw_in, mem_in, n, n)


def sycophancy_tokens(questions: int, rounds: int, answer_tokens: int) -> Line:
    """Full-history grows every round; memory condition is flat.

    The model's answers are unknown before the run, so each assistant reply is
    assumed to be ``answer_tokens`` long (the benchmark asks for short answers).
    """
    items = ITEMS[:questions]
    sys_t = estimate_tokens(SYSTEM_PROMPT)
    plain = _Plain()
    raw_in = mem_in = 0
    for item in items:
        q_t = estimate_tokens(item.question)
        push_t = estimate_tokens(pushback_text(plain, item.wrong, item.correct))
        # round 0 (same for both conditions): system + question
        raw_in += sys_t + q_t
        mem_in += sys_t + q_t
        # full history: system + q + a0 + r*(push + answer) ... minus the last answer
        for r in range(1, rounds + 1):
            raw_in += sys_t + q_t + answer_tokens + (r - 1) * (push_t + answer_tokens) + push_t
        # memory: static memory prompt + current push-back every round
        engine = build_memory_engine(
            state_dir=tempfile.mkdtemp(prefix="agent-memory-budget-syc-"),
            distiller=ConclusionDistiller(),
        )
        engine.process_turn(item.question, "x" * (answer_tokens * 4))
        ctx = engine.build_context(pushback_text(plain, item.wrong, item.correct))
        per_round = sum(estimate_tokens(m["content"]) for m in ctx.to_messages())
        mem_in += rounds * per_round
    n = len(items) * (rounds + 1)
    return Line("sycophancy", f"questions={questions} rounds={rounds}", 2 * n, raw_in, mem_in, n, n)


def price_line(line: Line, price: Tuple[float, float], out_tokens: int) -> Tuple[float, float]:
    """(raw_cost, memory_cost) in USD for one line under one model's price."""
    pin, pout = price
    raw = line.raw_in / 1e6 * pin + line.out_calls_raw * out_tokens / 1e6 * pout
    mem = line.mem_in / 1e6 * pin + line.out_calls_mem * out_tokens / 1e6 * pout
    return raw, mem


def parse_price(spec: str) -> Tuple[str, Tuple[float, float]]:
    model, _, rest = spec.partition("=")
    pin, _, pout = rest.partition(":")
    if not model or not pin:
        raise argparse.ArgumentTypeError("expected MODEL=IN[:OUT] in USD per million tokens")
    return model, (float(pin), float(pout or pin))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, nargs="+", default=[200, 1000, 5000],
                        help="context-rot transcript lengths to price (each is one setting)")
    parser.add_argument("--facts", type=int, default=10, help=f"facts per transcript (max {len(VALUE_POOL)})")
    parser.add_argument("--questions", type=int, default=len(ITEMS), help="sycophancy items")
    parser.add_argument("--rounds", type=int, default=4, help="push-back rounds per item")
    parser.add_argument("--seeds", type=int, default=5, help="independent repeats per setting (for error bars)")
    parser.add_argument("--out-tokens", type=int, default=40, help="assumed reply length per call")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_PRICES), help="models to price")
    parser.add_argument("--price", type=parse_price, action="append", default=[],
                        help="override/add a price: MODEL=IN:OUT ($ per million tokens)")
    args = parser.parse_args(argv)

    prices = dict(DEFAULT_PRICES)
    for model, p in args.price:
        prices[model] = p
    unknown = [m for m in args.models if m not in prices]
    if unknown:
        parser.error(f"no price for {unknown}; pass --price MODEL=IN:OUT")

    lines: List[Line] = []
    for turns in args.turns:
        lines.append(context_rot_tokens(turns, args.facts, seed=0))
    lines.append(sycophancy_tokens(args.questions, args.rounds, args.out_tokens))

    print(f"Campaign: {len(args.turns)} context-rot settings + 1 sycophancy setting, "
          f"x{args.seeds} seeds, x{len(args.models)} models\n")
    print("| Benchmark | Setting | Calls/seed | Raw input tok | Memory input tok | Memory share |")
    print("|---|---|---|---|---|---|")
    for ln in lines:
        share = ln.mem_in / max(1, ln.raw_in) * 100
        print(f"| {ln.benchmark} | {ln.setting} | {ln.calls} | {ln.raw_in:,} | {ln.mem_in:,} | {share:.1f}% |")

    total_calls = sum(ln.calls for ln in lines) * args.seeds
    total_in = sum(ln.total_in for ln in lines) * args.seeds
    print(f"\nPer model: {total_calls:,} calls, ≈{total_in:,} input tokens "
          f"(+ ≈{total_calls * args.out_tokens:,} output tokens assumed).\n")

    print("| Model | $/M in:out | Raw condition | Memory condition | Total (all seeds) |")
    print("|---|---|---|---|---|")
    grand = 0.0
    for model in args.models:
        raw_c = mem_c = 0.0
        for ln in lines:
            r, m = price_line(ln, prices[model], args.out_tokens)
            raw_c += r * args.seeds
            mem_c += m * args.seeds
        grand += raw_c + mem_c
        pin, pout = prices[model]
        print(f"| {model} | {pin:.2f}:{pout:.2f} | ${raw_c:,.2f} | ${mem_c:,.2f} | ${raw_c + mem_c:,.2f} |")
    print(f"\nWhole campaign, {len(args.models)} models: ≈ ${grand:,.2f}")
    print("(input-token estimates are exact for the prompts the harness builds; "
          "output and prices are assumptions — override with --out-tokens / --price)")


if __name__ == "__main__":
    main()
