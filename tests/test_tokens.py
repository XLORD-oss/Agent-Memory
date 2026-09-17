"""Cost-model tests: the quadratic-vs-linear arithmetic is the framework's core claim."""

from agent_memory.tokens import (
    compact_total_tokens,
    compare,
    cost_table,
    estimate_tokens,
    full_history_total_tokens,
)


def test_estimate_tokens_heuristics():
    assert estimate_tokens("") >= 1
    # ~4 chars/token heuristic: 1000 chars ≈ 250 tokens
    assert estimate_tokens("a" * 1000, method="chars4") == 250
    assert estimate_tokens("x", method="chars4") == 1


def test_full_history_is_quadratic():
    # Turn N re-sends N-1 prior turns -> total = per_turn * T*(T+1)/2
    assert full_history_total_tokens(1, 500) == 500
    assert full_history_total_tokens(2, 500) == 1500  # 500 + 1000
    assert full_history_total_tokens(3, 500) == 3000  # 500 + 1000 + 1500
    # Doubling T multiplies the bill ~4x (asymptotically quadratic): far more
    # than the 2x a linear policy would cost.
    assert full_history_total_tokens(2000, 500) > 3.5 * full_history_total_tokens(1000, 500)


def test_compact_is_near_linear():
    # Memory is capped, so doubling T should roughly double (linear) the cost,
    # far below the quadratic full-history bill.
    c1000 = compact_total_tokens(1000, 500, memory_cap=3000)
    c2000 = compact_total_tokens(2000, 500, memory_cap=3000)
    assert c2000 < 2.5 * c1000  # clearly sub-quadratic (4x would be quadratic)
    assert c2000 < full_history_total_tokens(2000, 500)


def test_compare_ratio():
    for turns in (100, 500, 1000, 2000):
        cmp = compare(turns=turns, per_turn=500)
        assert cmp.full_history > 0
        assert 0 < cmp.ratio < 1
        assert cmp.saved_fraction == 1 - cmp.ratio
        # The saving grows with conversation length.
    small = compare(turns=50)
    big = compare(turns=2000)
    assert big.saved_fraction > small.saved_fraction


def test_cost_table_is_markdown():
    table = cost_table(turn_counts=(50, 100))
    assert "Full history" in table
    assert "Compact memory" in table
    assert "|" in table
    lines = [l for l in table.splitlines() if l.strip()]
    assert len(lines) == 4  # header + separator + 2 data rows
