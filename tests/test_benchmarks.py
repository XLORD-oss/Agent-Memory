"""Offline tests for the benchmark harnesses using the MockModel.

The mock emulates documented failure modes deterministically, so these tests
verify the harness logic (two conditions, scoring, metrics) without an API key.
"""

import tempfile

from agent_memory.context import SYSTEM_PROMPT
from agent_memory.llm import MockModel

from benchmarks.common.harness import Trial, exact_match, make_client, render_question
from benchmarks.common.memory_builder import ConclusionDistiller, MarkerDistiller, build_memory_engine
from benchmarks.context_rot.tasks import generate_transcript, question_for
from benchmarks.sycophancy.run_flipflop import is_flip, run_item_full_history, run_item_memory
from benchmarks.sycophancy.tasks import ITEMS


class _Args:
    mock = True
    seed = 0
    model = "mock"
    base_url = None
    api_key_env = "OPENAI_API_KEY"


def test_context_rot_mock_shows_gap():
    client = make_client(_Args(), mock_mode="context_rot")
    transcript = generate_transcript(turns=80, n_facts=8, seed=1)

    engine = build_memory_engine(tempfile.mkdtemp(), distiller=MarkerDistiller())
    for line in transcript.raw.splitlines():
        engine.process_turn(line)
    mem_ctx = engine.build_context("")

    raw_correct = 0
    mem_correct = 0
    for fact in transcript.facts:
        q = render_question(client, fact.id, fact.value, "UNKNOWN", question_for(fact))
        raw_user = "TRANSCRIPT\n" + transcript.raw + "\n\nQUESTION\n" + q
        raw_resp = client.complete(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": raw_user}]
        )
        raw_correct += int(exact_match(raw_resp, fact.value))

        mem_user = mem_ctx.user_prompt + "\n\nQUESTION\n" + q
        mem_resp = client.complete(
            [{"role": "system", "content": mem_ctx.system}, {"role": "user", "content": mem_user}]
        )
        mem_correct += int(exact_match(mem_resp, fact.value))

    # Raw transcript suffers lost-in-the-middle; compact memory recovers it.
    assert mem_correct > raw_correct
    assert mem_correct == len(transcript.facts)


def test_sycophancy_mock_full_flips_memory_holds():
    client = MockModel(mode="sycophancy", seed=2)
    item = ITEMS[0]
    full = run_item_full_history(client, item, rounds=4, system=SYSTEM_PROMPT)
    mem = run_item_memory(client, item, rounds=4, system=SYSTEM_PROMPT)

    assert full["tof"] is not None  # full history flips under pressure
    assert mem["tof"] is None  # memory condition holds
    assert mem["nof"] == 0


def test_is_flip_detection():
    assert is_flip("You are right, it is London.", "Paris", "London")
    assert not is_flip("The answer is Paris.", "Paris", "London")
    assert not is_flip("I think you might be right.", "Paris", "London")


def test_marker_distiller_keeps_ground_truth():
    ext = MarkerDistiller().distill(
        "One thing to remember: [[FACT:3]] = Paris. Plus some chatter.", "", 7
    )
    assert len(ext.facts) == 1
    assert "[[FACT:3]] = Paris" in ext.facts[0].text


def test_trial_accuracy_helper():
    trials = [
        Trial(id=1, condition="raw", response="x", correct=False),
        Trial(id=2, condition="raw", response="y", correct=True),
        Trial(id=3, condition="memory", response="z", correct=True),
    ]
    from benchmarks.common.harness import accuracy

    assert accuracy(trials, "raw") == 0.5
    assert accuracy(trials, "memory") == 1.0
