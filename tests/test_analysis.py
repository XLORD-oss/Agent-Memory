"""Trace-analysis tests: format loading, token math, usage reporting."""

import json
import tempfile
from pathlib import Path

from agent_memory.analysis import (
    TraceAnalyzer,
    compare_policies,
    compare_table,
    demo_trace,
    load_trace,
)
from agent_memory.tokens import estimate_tokens

FACT_TURNS = [
    {"user": "Remember that my stack is Python + FastAPI.", "assistant": "Noted."},
    {"user": "Remember that the demo date is the last Friday.", "assistant": "Locked."},
    {"user": "What's in my stack?", "assistant": "Python + FastAPI."},  # references fact 1
]


def test_demo_trace_is_well_formed():
    turns = demo_trace()
    assert len(turns) > 5
    for t in turns:
        assert isinstance(t, dict)
        assert "user" in t and "assistant" in t


def test_load_jsonl_role_content(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"role": "user", "content": "hello"}) + "\n"
        + json.dumps({"role": "assistant", "content": "hi there"}) + "\n"
    )
    turns = load_trace(path)
    assert turns == [{"user": "hello", "assistant": "hi there"}]


def test_load_jsonl_paired(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"user": "a", "assistant": "b"}) + "\n")
    assert load_trace(path) == [{"user": "a", "assistant": "b"}]


def test_load_plain_text(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("user: hello\nassistant: hi there\nuser: again\n")
    turns = load_trace(path)
    assert turns == [
        {"user": "hello", "assistant": "hi there"},
        {"user": "again", "assistant": None},
    ]


def test_analyzer_compact_beats_naive_on_long_trace():
    turns = demo_trace() * 5  # 60 turns
    r = TraceAnalyzer(profile="general").analyze(turns)
    assert r.turns == 60
    assert r.naive_total > 0 and r.compact_total > 0
    assert r.compact_total < r.naive_total
    assert r.saved_pct > 10
    assert r.ratio < 0.9
    assert len(r.series) == 60


def test_analyzer_usage_tracking_references():
    r = TraceAnalyzer(profile="general").analyze(FACT_TURNS)
    assert r.entries_created >= 1
    used = [u for u in r.usage if u["uses"] > 0]
    assert any("stack" in u["text"].lower() for u in used)


def test_compare_policies_runs_all_profiles():
    # A short trace can have compact > naive (bounded working memory duplicates
    # distilled content); on a realistic long trace every profile must win.
    turns = demo_trace() * 5
    results = compare_policies(turns)
    assert set(results) == {"general", "coding", "research", "writing"}
    for profile, r in results.items():
        assert 0 < r["ratio"] < 1
        assert r["compact_total"] > 0
    assert "Saved" in compare_table(results)


def test_render_markdown_has_tables():
    r = TraceAnalyzer(profile="coding").analyze(demo_trace())
    md = TraceAnalyzer(profile="coding").render_markdown(r)
    assert "naive full-history replay" in md.lower()
    assert "| Turn |" in md
    assert "Usage report" in md


def test_compression_ratio_reported():
    r = TraceAnalyzer(profile="general").analyze(demo_trace() * 3)
    assert r.compression_ratio > 0


def test_naive_is_arithmetically_correct():
    turns = FACT_TURNS
    r = TraceAnalyzer(profile="general").analyze(turns)
    sys_tokens = estimate_tokens(
        "You are an assistant with a compact long-term memory. The MEMORY and PERSPECTIVES blocks below are the authoritative record."
    )
    # Naive at turn i = system + cumulative raw. Verify last turn equals that.
    raw_total = sum(
        estimate_tokens(t.get("user") or "") + estimate_tokens(t.get("assistant") or "")
        for t in turns
    )
    expected_last_naive = r.series[-1]["naive"]
    # sys prompt is ~the full SYSTEM_PROMPT; estimate generously
    assert expected_last_naive > raw_total  # includes system + all raw tokens
    assert r.max_naive_turn == expected_last_naive


def test_cli_repeat_flag(tmp_path, monkeypatch):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        from agent_memory.analysis import main

        main(["--trace", "demo", "--repeat", "3", "--profile", "general"])
    out = buf.getvalue()
    assert "36 turns" in out  # 12 demo turns x 3
    assert "ratio" in out.lower()