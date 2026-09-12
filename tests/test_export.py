"""SFT export: identical targets, four context contracts, honest token accounting."""

from __future__ import annotations

import json

import pytest

from agent_memory.analysis import demo_trace
from agent_memory.export import (
    CONDITIONS,
    NEUTRAL_SYSTEM,
    SFTExporter,
    main,
    messages_tokens,
    summarize,
    write_jsonl,
)


@pytest.fixture(scope="module")
def examples():
    return SFTExporter().export_trace(demo_trace(), trace_id="demo")


def _by(examples, cond):
    return sorted((e for e in examples if e.condition == cond), key=lambda e: e.turn)


def test_every_turn_yields_one_example_per_condition(examples):
    turns = sorted({e.turn for e in examples})
    assert turns == list(range(1, len(demo_trace()) + 1))
    for cond in CONDITIONS:
        assert [e.turn for e in _by(examples, cond)] == turns


def test_targets_are_identical_across_conditions(examples):
    for turn in {e.turn for e in examples}:
        comps = {e.condition: e.completion for e in examples if e.turn == turn}
        assert len({json.dumps(c) for c in comps.values()}) == 1
        assert comps["memory"][0]["role"] == "assistant"


def test_only_replay_conditions_expose_prior_assistant_output(examples):
    late = [e for e in examples if e.turn >= 3]
    assert all(e.has_prior_assistant for e in late if e.condition == "full")
    assert all(e.has_prior_assistant for e in late if e.condition == "matched")
    assert not any(e.has_prior_assistant for e in late if e.condition == "memory")
    assert not any(e.has_prior_assistant for e in late if e.condition == "user_only")


def test_prompts_end_with_current_user_message_and_share_system(examples):
    trace = demo_trace()
    for e in examples:
        assert e.prompt[0] == {"role": "system", "content": NEUTRAL_SYSTEM}
        last = e.prompt[-1]
        assert last["role"] == "user"
        if e.condition == "memory":
            assert trace[e.turn - 1]["user"] in last["content"]  # inside CURRENT USER MESSAGE
        else:
            assert last["content"] == trace[e.turn - 1]["user"]


def test_matched_control_is_length_matched_and_well_formed(examples):
    mem = {e.turn: e for e in _by(examples, "memory")}
    for e in _by(examples, "matched"):
        # never longer than the memory prompt (unless irreducible: system + current user only)
        assert e.prompt_tokens <= mem[e.turn].prompt_tokens or len(e.prompt) == 2
        assert e.prompt_tokens == messages_tokens(e.prompt)
        body = e.prompt[1:]
        assert body[0]["role"] == "user"  # no dangling assistant opener
        # alternation holds
        for a, b in zip(body, body[1:]):
            assert a["role"] != b["role"]


def test_full_grows_while_memory_stays_bounded(examples):
    full = _by(examples, "full")
    mem = _by(examples, "memory")
    assert full[-1].prompt_tokens > 2 * full[0].prompt_tokens
    assert max(e.prompt_tokens for e in mem) <= 3000  # inside the memory cap
    # memory condition is cheaper than full by the end of the conversation
    assert mem[-1].prompt_tokens < full[-1].prompt_tokens


def test_user_only_is_full_minus_assistant_messages(examples):
    full = {e.turn: e for e in _by(examples, "full")}
    for e in _by(examples, "user_only"):
        expected = [m for m in full[e.turn].prompt if m["role"] != "assistant"]
        assert e.prompt == expected


def test_summary_and_jsonl_roundtrip(tmp_path, examples):
    summary = summarize(examples)
    assert set(summary) == set(CONDITIONS)
    assert summary["full"]["share_with_prior_assistant"] > 0.8
    assert summary["memory"]["share_with_prior_assistant"] == 0.0
    paths = write_jsonl(examples, tmp_path)
    for cond in CONDITIONS:
        rows = [json.loads(l) for l in paths[cond].read_text().splitlines()]
        assert len(rows) == len(demo_trace())
        assert {"prompt", "completion", "condition", "turn"} <= set(rows[0])
    assert json.loads(paths["summary"].read_text())["memory"]["examples"] == len(demo_trace())


def test_cli_with_trace_file_and_condition_subset(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    trace.write_text("\n".join(json.dumps(t) for t in demo_trace()[:5]))
    main([str(trace), "--out", str(tmp_path / "out"), "--conditions", "memory", "full", "--system", "contract"])
    out = capsys.readouterr().out
    assert "5 targets × 2 conditions" in out
    assert (tmp_path / "out" / "train_memory.jsonl").exists()
    assert not (tmp_path / "out" / "train_matched.jsonl").exists()
    first = json.loads((tmp_path / "out" / "train_memory.jsonl").read_text().splitlines()[0])
    assert "compact long-term memory" in first["prompt"][0]["content"]


def test_rejects_unknown_condition():
    with pytest.raises(ValueError):
        SFTExporter(conditions=("memory", "telepathy"))
