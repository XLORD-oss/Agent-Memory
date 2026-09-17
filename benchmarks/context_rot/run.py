"""Context-rot benchmark: raw transcript vs. compact memory.

Adapts Chroma's focused-vs-full-prompt methodology (see docs/evidence.md) into a
direct test of this framework's central claim: for the SAME task and SAME model,
does replacing the full raw transcript with the compact memory file recover the
accuracy that long-context degradation eats?

    python -m benchmarks.context_rot.run --mock                 # offline demo
    python -m benchmarks.context_rot.run --model gpt-4o-mini    # real model
    python -m benchmarks.context_rot.run --help

Methodology
-----------
1. Generate a long noisy conversation (``--turns``) with ``--facts`` ground-truth
   facts planted at scattered positions (needles in a haystack).
2. Condition *raw*: ask the model each question with the FULL transcript in
   context — the naive default every agent ships with.
3. Condition *memory*: distill the transcript into the compact memory file and
   ask the same questions with only memory + a small rolling window.
4. Score exact-match accuracy per condition and report the gap.

The mock model emulates the documented lost-in-the-middle + length-rot behavior
so the harness runs offline; a real model replaces it with zero code changes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory.context import SYSTEM_PROMPT  # noqa: E402
from agent_memory.tokens import estimate_tokens  # noqa: E402

from benchmarks.common.harness import (  # noqa: E402
    Trial,
    accuracy,
    add_model_args,
    exact_match,
    is_mock,
    make_client,
    print_accuracy_table,
    prompt_stats,
    render_question,
    seed_plan,
    write_json,
)
from benchmarks.common.memory_builder import MarkerDistiller, build_memory_engine  # noqa: E402
from benchmarks.context_rot.tasks import generate_transcript, question_for  # noqa: E402


def raw_user_prompt(transcript_raw: str, q_text: str) -> str:
    """The full-transcript condition: everything so far, then the question."""
    return "TRANSCRIPT (the full conversation so far)\n" + transcript_raw + "\n\nQUESTION\n" + q_text


def memory_user_prompt(mem_ctx, q_text: str) -> str:
    """The compact-memory condition: the assembled memory prompt, then the question."""
    return mem_ctx.user_prompt + "\n\nQUESTION\n" + q_text


def run_once(client, turns: int, facts: int, seed: int, verbose: bool = True) -> dict:
    """One seed: build the transcript + memory, ask every question under both conditions."""
    transcript = generate_transcript(turns, facts, seed)

    # --- build the compact-memory condition --------------------------------
    engine = build_memory_engine(
        state_dir=tempfile.mkdtemp(prefix="agent-memory-bench-"),
        distiller=MarkerDistiller(),
    )
    for line in transcript.raw.splitlines():
        engine.process_turn(line)
    mem_ctx = engine.build_context("")
    print(
        f"[setup] transcript={transcript.turns} turns, {transcript.n_facts} facts; "
        f"memory file = {engine.memory_tokens()} tokens vs transcript ≈ "
        f"{estimate_tokens(transcript.raw)} tokens "
        f"({estimate_tokens(transcript.raw) / max(1, engine.memory_tokens()):.0f}x smaller)"
    )

    # --- run both conditions over the same questions ------------------------
    trials: list[Trial] = []
    input_tokens = {"raw": 0, "memory": 0}
    per_call: dict = {"raw": [], "memory": []}  # measured prompt size per question, per arm

    for fact in transcript.facts:
        q = question_for(fact)
        q_text = render_question(client, fact.id, fact.value, "UNKNOWN", q)

        raw_user = raw_user_prompt(transcript.raw, q_text)
        raw_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": raw_user}]
        raw_resp = client.complete(raw_messages)
        raw_ps = prompt_stats(raw_messages)
        input_tokens["raw"] += raw_ps["prompt_tokens"]
        per_call["raw"].append(raw_ps["prompt_tokens"])
        trials.append(
            Trial(
                id=fact.id,
                condition="raw",
                response=raw_resp,
                correct=exact_match(raw_resp, fact.value),
                metadata={"fact_turn": fact.turn, "value": fact.value,
                          "prompt_tokens": raw_ps["prompt_tokens"],
                          "completion_tokens": estimate_tokens(raw_resp),
                          "fact_relative_position": round(fact.turn / max(1, transcript.turns), 3)},
            )
        )

        mem_user = memory_user_prompt(mem_ctx, q_text)
        mem_messages = [{"role": "system", "content": mem_ctx.system}, {"role": "user", "content": mem_user}]
        mem_resp = client.complete(mem_messages)
        mem_ps = prompt_stats(mem_messages)
        input_tokens["memory"] += mem_ps["prompt_tokens"]
        per_call["memory"].append(mem_ps["prompt_tokens"])
        trials.append(
            Trial(
                id=fact.id,
                condition="memory",
                response=mem_resp,
                correct=exact_match(mem_resp, fact.value),
                metadata={"fact_turn": fact.turn, "value": fact.value,
                          "prompt_tokens": mem_ps["prompt_tokens"],
                          "completion_tokens": estimate_tokens(mem_resp),
                          "fact_relative_position": round(fact.turn / max(1, transcript.turns), 3)},
            )
        )

        if verbose and is_mock(client) and client.last_stats:
            print(f"  fact {fact.id:>2} raw pos={client.last_stats.get('relative_pos')} "
                  f"len={client.last_stats.get('length_tokens')} found={client.last_stats.get('found')} "
                  f"-> {'hit' if trials[-2].correct else 'miss'}")

    # --- report ------------------------------------------------------------
    print_accuracy_table(trials, ["raw", "memory"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    print("\nWhat each arm was actually handed (measured):")
    print("| Arm | Input tok / question (mean) | min | max | Total input tok |")
    print("|---|---|---|---|---|")
    for arm in ("raw", "memory"):
        xs = per_call[arm]
        print(f"| {arm} | {mean(xs):,.0f} | {min(xs):,} | {max(xs):,} | {input_tokens[arm]:,} |")
    ratio = mean(per_call["memory"]) / max(1e-9, mean(per_call["raw"]))
    print(f"  memory / raw input ratio: {ratio:.3f}  (the length confound: accuracy gaps below are between prompts "
          f"{1 / max(ratio, 1e-9):.0f}× apart in size — not a like-for-like comparison of *content*)")

    return {
        "transcript_turns": transcript.turns,
        "n_facts": transcript.n_facts,
        "summary": {
            "raw_accuracy": accuracy(trials, "raw"),
            "memory_accuracy": accuracy(trials, "memory"),
            "raw_prompt_tokens_mean": mean(per_call["raw"]),
            "memory_prompt_tokens_mean": mean(per_call["memory"]),
            "memory_to_raw_ratio": ratio,
        },
        "input_tokens": input_tokens,
        "trials": [t.to_dict() for t in trials],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--turns", type=int, default=200, help="conversation length in turns")
    parser.add_argument("--facts", type=int, default=10, help="number of ground-truth facts")
    args = parser.parse_args()

    plan = seed_plan(args, "context_rot")
    if not plan:
        print("nothing to do — every seed already has a result on disk")
        return

    client = None
    for seed, out_path in plan:
        args.seed = seed
        if client is None or is_mock(client):
            client = make_client(args, mock_mode="context_rot")
        if len(plan) > 1:
            print(f"\n=== seed {seed} ===")
        payload = run_once(client, args.turns, args.facts, seed, verbose=len(plan) == 1)
        write_json(out_path, {"args": vars(args), "seed": seed, **payload})


if __name__ == "__main__":
    main()
