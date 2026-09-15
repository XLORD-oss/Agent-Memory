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
