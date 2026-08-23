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

from agent_memory.llm import Client, MockModel, OpenAICompatClient  # noqa: E402


def add_model_args(parser: argparse.ArgumentParser, default_model: str = "gpt-4o-mini") -> None:
    parser.add_argument("--model", default=default_model, help="model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the offline MockModel emulating documented failure modes",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--out",
        default=None,
        help="path to write JSON results (created under runs/ by default)",
    )


def make_client(args: argparse.Namespace, mock_mode: str = "context_rot") -> Client:
    """Build a MockModel or an OpenAI-compatible client from CLI args."""
    if args.mock:
        return MockModel(mode=mock_mode, seed=args.seed)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"[harness] {args.api_key_env} is not set and --mock was not passed; "
            "falling back to the MockModel so the harness still runs.",
            file=sys.stderr,
        )
        return MockModel(mode=mock_mode, seed=args.seed)
    return OpenAICompatClient(model=args.model, base_url=args.base_url, api_key=api_key)


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
