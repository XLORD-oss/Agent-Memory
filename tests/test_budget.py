"""The budget estimator prices the exact prompts the benchmarks would send."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from benchmarks.budget import (
    DEFAULT_PRICES,
    context_rot_tokens,
    main,
    parse_price,
    price_line,
    sycophancy_tokens,
)
from benchmarks.sycophancy.tasks import ITEMS


def test_context_rot_memory_condition_is_flat_while_raw_grows():
    short = context_rot_tokens(turns=100, facts=5, seed=0)
    long = context_rot_tokens(turns=800, facts=5, seed=0)
    assert short.calls == long.calls == 10  # 2 conditions x 5 facts
    assert long.raw_in > 5 * short.raw_in  # raw scales with transcript length
    assert abs(long.mem_in - short.mem_in) < 0.25 * short.mem_in  # memory does not
    assert long.mem_in < 0.1 * long.raw_in


def test_sycophancy_accounting_matches_harness_call_pattern():
    rounds, questions, answer = 3, 4, 40
    line = sycophancy_tokens(questions=questions, rounds=rounds, answer_tokens=answer)
    # each item: 1 initial call + `rounds` push-back calls, in both conditions
    assert line.calls == 2 * questions * (rounds + 1)
    assert line.out_calls_raw == line.out_calls_mem == questions * (rounds + 1)
    # full-history accumulates prior answers; memory replays none of them
    assert line.raw_in > 0 and line.mem_in > 0
    # sanity: more rounds -> strictly more raw tokens, growth faster than memory
    more = sycophancy_tokens(questions=questions, rounds=rounds + 3, answer_tokens=answer)
    assert (more.raw_in - line.raw_in) > (more.mem_in - line.mem_in)


def test_price_line_is_linear_in_tokens_and_output_assumption():
    line = context_rot_tokens(turns=100, facts=4, seed=1)
    raw_a, mem_a = price_line(line, (1.0, 1.0), out_tokens=0)
    assert abs(raw_a - line.raw_in / 1e6) < 1e-12
    assert abs(mem_a - line.mem_in / 1e6) < 1e-12
    raw_b, _ = price_line(line, (1.0, 1.0), out_tokens=100)
    assert abs((raw_b - raw_a) - line.out_calls_raw * 100 / 1e6) < 1e-12


def test_parse_price_accepts_in_only_and_in_out():
    assert parse_price("m=2.5") == ("m", (2.5, 2.5))
    assert parse_price("vendor/model=0.1:0.4") == ("vendor/model", (0.1, 0.4))


def test_cli_runs_offline_and_reports_every_model():
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--turns", "50", "120", "--facts", "4", "--seeds", "2",
              "--questions", "2", "--rounds", "2",
              "--price", "local/test=1.00:2.00", "--models", "local/test", "openai/gpt-4o-mini"])
    out = buf.getvalue()
    assert "context_rot" in out and "sycophancy" in out
    assert "local/test" in out and "openai/gpt-4o-mini" in out
    assert "Whole campaign, 2 models" in out
    assert len(ITEMS) >= 2 and all(m in DEFAULT_PRICES for m in ["openai/gpt-4o"])
