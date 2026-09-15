"""Shared benchmark harness: model client construction, scoring, reporting.

The same code paths run against the offline ``MockModel`` (for CI and to
demonstrate the methodology) or any OpenAI-compatible real model. The mock
emulates the *documented* failure modes (context rot, lost-in-the-middle,
sycophantic flipping) so the harness itself is testable without an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the installed package importable when running from a source checkout.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_memory.llm import Client, LocalHFClient, MockModel, OpenAICompatClient, score_record  # noqa: E402


def add_model_args(parser: argparse.ArgumentParser, default_model: str = "gpt-4o-mini") -> None:
    parser.add_argument("--model", default=default_model, help="model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (takes priority over --api-key-env)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key (used only if --api-key is not set)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the offline MockModel emulating documented failure modes",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run --model in-process with transformers (LocalHFClient); "
             "for notebooks/GPUs without a server. Needs torch+transformers.",
    )
    parser.add_argument("--dtype", default="float16", help="dtype for --local (T4 has no bfloat16)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="reply length cap for --local")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (single run)")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="run seeds 0..N-1 in one process, one JSON per seed, skipping seeds already on disk "
             "(resumable across session limits)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="path to write JSON results (created under runs/ by default; with --seeds, a directory)",
    )


def make_client(args: argparse.Namespace, mock_mode: str = "context_rot") -> Client:
    """Build a MockModel or an OpenAI-compatible client from CLI args."""
    if args.mock:
        args.model = "mock"  # label results honestly
        return MockModel(mode=mock_mode, seed=args.seed)
    if getattr(args, "local", False):
        return LocalHFClient(
            args.model, dtype=getattr(args, "dtype", "float16"),
            max_new_tokens=getattr(args, "max_new_tokens", 64),
        )
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"[harness] no API key provided (--api-key or {args.api_key_env}) "
            "and --mock was not passed; "
            "falling back to the MockModel so the harness still runs.",
            file=sys.stderr,
        )
        return MockModel(mode=mock_mode, seed=args.seed)
    return OpenAICompatClient(model=args.model, base_url=args.base_url, api_key=api_key)


def complete_scored(client: Client, messages: List[dict], **kwargs: Any) -> Dict[str, Any]:
    """Get a reply plus confidence when the backend can score it.

    Backends with ``complete_scored`` (LocalHFClient, OpenAICompatClient against
    vLLM/OpenAI) return token logprobs; anything else (MockModel, servers that
    refuse ``logprobs``) degrades to text-only with ``seq_confidence=None``.
    """
    fn = getattr(client, "complete_scored", None)
    if fn is not None:
        try:
            return fn(messages, **kwargs)
        except Exception as exc:  # server rejected logprobs, etc. -> text only
            print(f"[harness] scored completion unavailable ({exc.__class__.__name__}); falling back to text", file=sys.stderr)
    return score_record(client.complete(messages, **kwargs))


def seed_plan(args: argparse.Namespace, name: str) -> List[tuple]:
    """(seed, out_path) pairs to run; seeds whose output already exists are skipped.

    Single run: ``[(args.seed, args.out or runs/<name>.json)]``. With ``--seeds N``:
    ``runs/<name>/seed_<k>.json`` for k in 0..N-1, minus the ones on disk — so a
    campaign interrupted by a session limit resumes where it stopped.
    """
    if getattr(args, "seeds", None) is None:
        return [(args.seed, Path(args.out) if args.out else default_out_path(name))]
    base = Path(args.out) if args.out else default_out_path(name).with_suffix("")
    plan = []
    for k in range(args.seeds):
        path = base / f"seed_{k}.json"
        if path.exists():
            print(f"[harness] seed {k} already done -> {path} (skip)")
            continue
        plan.append((k, path))
    return plan


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def exact_match(response: str, correct: str) -> bool:
    """True if the normalized correct value appears in the response."""
    return normalize(correct) in normalize(response)


def question_block(qid: int, correct: str, wrong: str) -> str:
    """The annotated question block both mock and real paths understand."""
    return (
        f"[[QUESTION:{qid}]]\n"
        f"[[CORRECT:{correct}]]\n"
        f"[[WRONG:{wrong}]]"
    )


def is_mock(client: Client) -> bool:
    return isinstance(client, MockModel)


def render_question(
    client: Client, qid: int, correct: str, wrong: str, plain_question: str
) -> str:
    """Build the question text sent to the model.

    The mock reads the ``[[CORRECT:...]]`` / ``[[WRONG:...]]`` annotations to
    emulate failure modes deterministically; real models must never see them, so
    they are stripped for every non-mock client.
    """
    if is_mock(client):
        return f"{question_block(qid, correct, wrong)}\n{plain_question}"
    return plain_question


@dataclass
class Trial:
    id: int
    condition: str
    response: str
    correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "condition": self.condition,
            "response": self.response,
            "correct": self.correct,
            "metadata": self.metadata,
        }


def accuracy(trials: List[Trial], condition: str) -> float:
    subset = [t for t in trials if t.condition == condition]
    if not subset:
        return float("nan")
    return sum(t.correct for t in subset) / len(subset)


def print_accuracy_table(trials: List[Trial], conditions: List[str]) -> None:
    print()
    print("| Condition | Correct | Total | Accuracy |")
    print("|---|---|---|---|")
    for c in conditions:
        subset = [t for t in trials if t.condition == c]
        correct = sum(t.correct for t in subset)
        acc = (correct / len(subset)) if subset else float("nan")
        print(f"| {c} | {correct} | {len(subset)} | {acc * 100:.1f}% |")
    if len(conditions) == 2:
        a = accuracy(trials, conditions[0])
        b = accuracy(trials, conditions[1])
        gap = b - a
        print(f"\nGap ({conditions[1]} - {conditions[0]}): {gap * 100:+.1f} points")
        print(
            f"Interpretation: {conditions[1]} runs at {b * 100:.1f}% vs "
            f"{conditions[0]} at {a * 100:.1f}% — "
            f"{'an improvement of' if gap > 0 else 'a change of'} {abs(gap) * 100:.1f} points"
        )


def write_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


def default_out_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / f"{name}.json"
