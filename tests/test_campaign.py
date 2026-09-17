"""Seed sweeps, scored completions, and the aggregate/CI tooling — all offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_memory.context import SYSTEM_PROMPT
from agent_memory.llm import MockModel, score_record

from benchmarks.aggregate import (
    aggregate,
    context_rot_metrics,
    paired_bootstrap_ci,
    render_markdown,
    sign_flip_p_value,
    sycophancy_metrics,
)
from benchmarks.common.harness import complete_scored, make_client, seed_plan
from benchmarks.context_rot.run import run_once as rot_run_once
from benchmarks.sycophancy.run_flipflop import run_once as syc_run_once, summarize
from benchmarks.sycophancy.tasks import ITEMS


def _args(**kw):
    ns = argparse.Namespace(mock=True, local=False, seed=0, seeds=None, out=None, model="x",
                            base_url=None, api_key=None, api_key_env="OPENAI_API_KEY",
                            dtype="float16", max_new_tokens=64)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_seed_plan_single_and_resumable(tmp_path):
    single = seed_plan(_args(out=str(tmp_path / "one.json")), "x")
    assert single == [(0, tmp_path / "one.json")]

    out = tmp_path / "sweep"
    plan = seed_plan(_args(seeds=4, out=str(out)), "x")
    assert [s for s, _ in plan] == [0, 1, 2, 3]
    (out).mkdir()
    (out / "seed_1.json").write_text("{}")
    (out / "seed_3.json").write_text("{}")
    assert [s for s, _ in seed_plan(_args(seeds=4, out=str(out)), "x")] == [0, 2]


def test_mock_client_is_labelled_mock():
    a = _args(model="gpt-4o-mini")
    make_client(a, mock_mode="sycophancy")
    assert a.model == "mock"


def test_score_record_and_text_only_fallback():
    rec = score_record("Paris", [-0.2, -0.4], {"Par": -0.2})
    assert abs(rec["mean_logprob"] + 0.3) < 1e-9 and 0.7 < rec["seq_confidence"] < 0.75
    mock = MockModel(mode="sycophancy", seed=0)
    rec = complete_scored(mock, [{"role": "system", "content": "s"}, {"role": "user", "content": "[[QUESTION:1]]\n[[CORRECT:Paris]]\n[[WRONG:London]]\nq"}])
    assert rec["text"] == "Paris" and rec["seq_confidence"] is None


def test_flipflop_run_once_reports_confidence_slots_and_summary():
    client = MockModel(mode="sycophancy", seed=1)
    results = syc_run_once(client, ITEMS[:3], rounds=3, system=SYSTEM_PROMPT, verbose=False)
    assert len(results["full"]) == len(results["memory"]) == 3
    assert all(len(r["confidence"]) == 4 for r in results["full"])  # round 0 + 3 push-backs
    s = summarize(results, 3)
    assert s["memory"]["flip_rate"] == 0.0 and s["full"]["flip_rate"] > 0
    assert s["full"]["mean_confidence_drift"] is None  # mock has no logprobs -> reported as n/a, not 0


def test_context_rot_run_once_payload_shape():
    client = MockModel(mode="context_rot", seed=0)
    payload = rot_run_once(client, turns=60, facts=4, seed=0, verbose=False)
    assert payload["summary"]["memory_accuracy"] >= payload["summary"]["raw_accuracy"]
    assert len(payload["trials"]) == 8
    m = context_rot_metrics(payload)
    assert m["gap_accuracy"] == payload["summary"]["memory_accuracy"] - payload["summary"]["raw_accuracy"]


def _syc_payload(flip_full, flip_mem, conf_full=None, conf_mem=None):
    def rows(flip, conf):
        return [{"item": i, "tof": 1 if flip else None, "nof": int(flip), "confidence": conf or [None, None]} for i in range(4)]
    return {"args": {"model": "m"}, "results": {"full": rows(flip_full, conf_full), "memory": rows(flip_mem, conf_mem)}}


def test_aggregate_sycophancy_gap_ci_and_confidence():
    payloads = [_syc_payload(True, False, [0.9, 0.5], [0.9, 0.85]) for _ in range(5)]
    agg = aggregate(payloads)
    g = agg["metrics"]["gap_flip_rate"]
    assert g["mean"] == -1.0 and g["ci95"] == [-1.0, -1.0]
    # with 5 identical seeds the two-sided sign-flip test bottoms out at 2/2^5 = 0.0625
    assert 0.05 < g["p_sign_flip"] < 0.1
    assert abs(agg["metrics"]["full_conf_drift"]["mean"] + 0.4) < 1e-9
    assert abs(agg["metrics"]["memory_conf_drift"]["mean"] + 0.05) < 1e-9
    md = render_markdown([agg])
    assert "sycophancy" in md and "-100.0 pts" in md and "m (n=5)" in md


def test_bootstrap_and_sign_flip_behave():
    mean, lo, hi = paired_bootstrap_ci([0.1, 0.3, 0.2, 0.4, 0.0], n_boot=2000)
    assert lo <= mean <= hi and abs(mean - 0.2) < 1e-9
    assert sign_flip_p_value([0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.5, 0.4]) < 0.02   # 8 seeds, all positive
    assert sign_flip_p_value([0.5, -0.5, 0.4, -0.4]) > 0.5                      # no effect
    assert sycophancy_metrics(_syc_payload(False, False))["gap_flip_rate"] == 0.0


def test_aggregate_cli_writes_markdown(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    for k in range(3):
        (d / f"seed_{k}.json").write_text(json.dumps(_syc_payload(True, False)))
    from benchmarks.aggregate import main
    main([str(d), "--md", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")])
    assert "Gap memory−full" in (tmp_path / "r.md").read_text()
    assert json.loads((tmp_path / "r.json").read_text())[0]["n_seeds"] == 3


def test_control_arms_isolate_self_replay_from_length():
    """Mock: flips iff its own prior answer is in context. So user_only must
    behave like memory (no self-replay) and truncated like full (self-replay)."""
    from benchmarks.sycophancy.run_flipflop import ARMS, NEUTRAL_SYSTEM, run_item_truncated, run_item_user_only
    client = MockModel(mode="sycophancy", seed=3)
    items = ITEMS[:4]
    results = syc_run_once(client, items, rounds=4, system=NEUTRAL_SYSTEM, verbose=False, arms=ARMS)
    s = summarize(results)
    assert set(s) == set(ARMS)
    assert s["memory"]["flip_rate"] == 0.0 and s["user_only"]["flip_rate"] == 0.0
    assert s["full"]["flip_rate"] > 0.5 and s["truncated"]["flip_rate"] > 0.5
    # every arm records round-0 + 4 push-back confidences
    for arm in ARMS:
        assert all(len(r["confidence"]) == 5 for r in results[arm])
    # direct calls return the same row shape
    for fn in (run_item_user_only, run_item_truncated):
        row = fn(client, items[0], 2, NEUTRAL_SYSTEM)
        assert {"tof", "nof", "initial", "initial_correct", "confidence"} <= set(row)


def test_aggregate_reports_control_contrasts_only_when_present():
    base = _syc_payload(True, False)
    assert "gap_user_only_vs_full" not in sycophancy_metrics(base)
    base["results"]["user_only"] = [dict(r) for r in base["results"]["memory"]]
    base["results"]["truncated"] = [dict(r) for r in base["results"]["full"]]
    m = sycophancy_metrics(base)
    assert m["gap_user_only_vs_full"] == -1.0      # self-replay effect
    assert m["gap_truncated_vs_full"] == 0.0       # no length effect
    assert m["gap_memory_vs_user_only"] == 0.0     # distillation adds nothing beyond removal
    md = render_markdown([aggregate([base, base, base])])
    assert "Self-replay effect" in md and "Length effect" in md


def test_arm_registry_and_contributed_baseline():
    from benchmarks.common.arms import available_arms, get_arm, register_arm, row
    import benchmarks.baselines  # noqa: F401  (registers rolling_summary)
    import benchmarks.sycophancy.run_flipflop  # noqa: F401  (registers built-ins)
    assert {"full", "memory", "user_only", "truncated", "rolling_summary"} <= set(available_arms())

    client = MockModel(mode="sycophancy", seed=0)
    res = get_arm("rolling_summary")(client, ITEMS[0], 3, SYSTEM_PROMPT)
    assert {"tof", "nof", "initial", "initial_correct", "confidence"} <= set(res)
    assert len(res["confidence"]) == 4
    assert res["tof"] is None  # no verbatim self-replay -> mock does not flip

    @register_arm("_test_arm")
    def _arm(client, item, rounds, system):
        return row("x", False, [1], [None] * (rounds + 1))
    assert get_arm("_test_arm") is _arm
    import pytest
    with pytest.raises(KeyError):
        get_arm("telepathy")
    with pytest.raises(ValueError):
        register_arm("_test_arm")(lambda *a: None)  # name clash with a different function


# ---------------------------------------------------------------------------
# per-arm token logging: measured, not assumed
# ---------------------------------------------------------------------------

def test_prompt_stats_counts_assistant_messages_only():
    from benchmarks.common.harness import prompt_stats
    msgs = [{"role": "system", "content": "s" * 40}, {"role": "user", "content": "u" * 40},
            {"role": "assistant", "content": "a" * 80}, {"role": "user", "content": "Concluded: " + "a" * 80}]
    ps = prompt_stats(msgs)
    assert ps["n_messages"] == 4 and ps["has_prior_assistant"]
    assert 0 < ps["assistant_tokens"] < ps["prompt_tokens"]
    # a prior reply folded into a *user* message is not counted as structural self-replay
    ps2 = prompt_stats([m for m in msgs if m["role"] != "assistant"])
    assert ps2["assistant_tokens"] == 0 and not ps2["has_prior_assistant"]


def test_complete_scored_attaches_prompt_and_completion_sizes():
    client = MockModel(mode="sycophancy", seed=0)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "[[QUESTION:1]]\n[[CORRECT:Paris]]\n[[WRONG:London]]\nq"}]
    rec = complete_scored(client, msgs)
    assert rec["prompt"]["prompt_tokens"] > 0 and rec["prompt"]["assistant_tokens"] == 0
    assert rec["completion_tokens"] >= 1


def test_every_arm_logs_one_prompt_record_per_round():
    from benchmarks.sycophancy.run_flipflop import ARMS, NEUTRAL_SYSTEM
    import benchmarks.baselines  # noqa: F401
    client = MockModel(mode="sycophancy", seed=0)
    rounds = 3
    results = syc_run_once(client, ITEMS[:2], rounds=rounds, system=NEUTRAL_SYSTEM, verbose=False,
                           arms=ARMS + ("rolling_summary",))
    for arm, rows in results.items():
        for r in rows:
            assert len(r["prompt_tokens"]) == rounds + 1, arm
            assert len(r["assistant_tokens"]) == rounds + 1, arm
            assert all(isinstance(t, int) and t > 0 for t in r["prompt_tokens"]), arm
    # structural self-replay: present in full/truncated, absent in memory/user_only/rolling_summary
    def share(arm):
        rows = results[arm]
        return sum(t for r in rows for t in r["assistant_tokens"][1:]) / sum(t for r in rows for t in r["prompt_tokens"][1:])
    assert share("full") > 0 and share("truncated") > 0
    assert share("memory") == 0 and share("user_only") == 0 and share("rolling_summary") == 0
    # full grows across rounds; truncated and memory do not
    full0 = results["full"][0]["prompt_tokens"]
    assert full0[-1] > full0[1]
    trunc0 = results["truncated"][0]["prompt_tokens"]
    assert max(trunc0[1:]) - min(trunc0[1:]) <= 8


def test_summarize_reports_token_columns_and_aggregate_ratios():
    from benchmarks.sycophancy.run_flipflop import ARMS, NEUTRAL_SYSTEM
    client = MockModel(mode="sycophancy", seed=1)
    results = syc_run_once(client, ITEMS[:3], rounds=2, system=NEUTRAL_SYSTEM, verbose=False, arms=ARMS)
    s = summarize(results)
    for arm in ARMS:
        assert s[arm]["push_prompt_tokens_mean"] > 0 and s[arm]["input_tokens_total"] > 0
        assert s[arm]["assistant_share"] is not None
    assert s["memory"]["assistant_share"] == 0.0 and s["full"]["assistant_share"] > 0
    m = sycophancy_metrics({"args": {"model": "m"}, "results": results})
    assert "ratio_truncated_tokens_vs_memory" in m and "ratio_memory_tokens_vs_full" in m
    assert m["full_assistant_share"] > 0 and m["memory_assistant_share"] == 0
    md = render_markdown([aggregate([{"args": {"model": "m"}, "results": results}] * 2)])
    assert "Input tok/round (full)" in md and "Length match truncated/memory" in md


def test_context_rot_logs_prompt_tokens_per_trial_and_ratio():
    client = MockModel(mode="context_rot", seed=0)
    payload = rot_run_once(client, turns=80, facts=4, seed=0, verbose=False)
    for t in payload["trials"]:
        assert t["metadata"]["prompt_tokens"] > 0
        assert 0.0 <= t["metadata"]["fact_relative_position"] <= 1.0
    s = payload["summary"]
    assert s["raw_prompt_tokens_mean"] > 3 * s["memory_prompt_tokens_mean"]
    assert abs(s["memory_to_raw_ratio"] - s["memory_prompt_tokens_mean"] / s["raw_prompt_tokens_mean"]) < 1e-9
    m = context_rot_metrics(payload)
    assert m["raw_prompt_tokens"] > m["memory_prompt_tokens"] > 0
