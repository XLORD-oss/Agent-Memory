"""Export paired fine-tuning data: the same targets under different context contracts.

The question this answers: *is the memory contract a training variable?*

Every chat model today is trained on ``(full transcript → next reply)``. This
module rebuilds a chat trace into four training sets that share **identical
targets** and differ only in what the model is conditioned on — a 2×2 design:

======================  ==========================  ============================
                        long context                short context
======================  ==========================  ============================
replays own outputs     ``full``  (today's default)  ``matched`` (recency-truncated
                                                    to the memory prompt's length)
withholds own outputs   ``user_only`` (assistant     ``memory`` (the compact,
                        turns dropped)              distilled state)
======================  ==========================  ============================

Fine-tune the same base model on each, then evaluate every model under every
*test-time* format (the cross-eval matrix). If a ``memory``-trained model still
resists push-back when handed a full transcript at test time, the effect lives
in the weights; if not, it is a prompt-format effect — which is what the
inference-time framework already provides. Either answer is informative.

Output format is TRL / Axolotl-compatible conversational prompt-completion
JSONL::

    {"prompt": [{"role": "system", ...}, ..., {"role": "user", ...}],
     "completion": [{"role": "assistant", "content": "<target>"}],
     "condition": "memory", "turn": 7, "prompt_tokens": 412, ...}

Loss must be computed on ``completion`` only (TRL does this by default for
prompt-completion data). Training on the whole sequence would teach the
``full`` model to reproduce its *earlier* replies too, which is a confound.

CLI::

    python -m agent_memory.export trace.jsonl [more.jsonl ...] --out data/sft
    python -m agent_memory.export --demo --out /tmp/sft --profile general
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .analysis import demo_trace, load_trace
from .context import SYSTEM_PROMPT
from .core import MemoryEngine
from .distiller import DistillerLike
from .policy import MemoryPolicy
from .tokens import estimate_tokens

CONDITIONS = ("full", "user_only", "memory", "matched")

# Held constant across all four conditions by default so the system prompt is
# not a factor. ``--system contract`` applies the memory contract prompt to
# every condition instead (it contains anti-sycophancy instructions, which is a
# separate variable worth toggling deliberately, not by accident).
NEUTRAL_SYSTEM = "You are a helpful, truthful assistant."
SYSTEM_CHOICES = {"neutral": NEUTRAL_SYSTEM, "contract": SYSTEM_PROMPT}

_PER_MESSAGE_OVERHEAD = 4  # role/separator tokens, OpenAI-style accounting


def messages_tokens(messages: Sequence[dict]) -> int:
    return sum(estimate_tokens(m.get("content") or "") + _PER_MESSAGE_OVERHEAD for m in messages)


@dataclass
class Example:
    condition: str
    turn: int
    prompt: List[dict]
    completion: List[dict]
    prompt_tokens: int
    completion_tokens: int
    has_prior_assistant: bool
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "condition": self.condition,
            "turn": self.turn,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "has_prior_assistant": self.has_prior_assistant,
            "meta": self.meta,
        }


def _has_prior_assistant(prompt: Sequence[dict]) -> bool:
    return any(m.get("role") == "assistant" for m in prompt)


def _truncate_to_budget(messages: List[dict], budget: int) -> List[dict]:
    """Drop the oldest non-system messages until the prompt fits ``budget`` tokens.

    Always keeps the system message and the final (current) user message, so the
    control is "same length, still replays own outputs" — never an empty prompt.
    """
    system = [m for m in messages[:1] if m.get("role") == "system"]
    body = messages[len(system):]
    while len(body) > 1 and messages_tokens(system + body) > budget:
        body = body[1:]
    # Chat templates expect user/assistant alternation starting with a user
    # message; never open the body with a dangling assistant reply.
    while len(body) > 1 and body[0].get("role") == "assistant":
        body = body[1:]
    return system + body


class SFTExporter:
    """Turn one conversation trace into paired examples for each condition."""

    def __init__(
        self,
        conditions: Sequence[str] = CONDITIONS,
        profile: str = "general",
        policy: Optional[MemoryPolicy] = None,
        system: str = NEUTRAL_SYSTEM,
        distiller: Optional[DistillerLike] = None,
        memory_cap_tokens: int = 3000,
        recent_window: int = 4,
        min_turn: int = 1,
    ) -> None:
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"unknown conditions {unknown}; choose from {CONDITIONS}")
        self.conditions = tuple(conditions)
        self.profile = profile
        self.policy = policy
        self.system = system
        self.distiller = distiller
        self.memory_cap_tokens = memory_cap_tokens
        self.recent_window = recent_window
        self.min_turn = min_turn

    # -- one conversation ----------------------------------------------------
    def export_trace(self, turns: List[dict], trace_id: str = "trace", state_dir: Optional[str] = None) -> List[Example]:
        engine = MemoryEngine(
            state_dir=state_dir or tempfile.mkdtemp(prefix="agent-memory-sft-"),
            distiller=self.distiller,
            memory_cap_tokens=self.memory_cap_tokens,
            recent_window=self.recent_window,
        )
        history: List[dict] = [{"role": "system", "content": self.system}]
        examples: List[Example] = []

        for i, t in enumerate(turns, start=1):
            user = (t.get("user") or "").strip()
            assistant = (t.get("assistant") or "").strip()

            if user and assistant and i >= self.min_turn:
                completion = [{"role": "assistant", "content": assistant}]
                ctoks = estimate_tokens(assistant)
                meta = {"trace": trace_id, "memory_entries": engine.entry_count}
                built: Dict[str, List[dict]] = {}

                full_prompt = history + [{"role": "user", "content": user}]
                built["full"] = full_prompt
                built["user_only"] = [m for m in full_prompt if m.get("role") != "assistant"]

                ctx = engine.build_context(user, profile=self.profile, policy=self.policy)
                mem_prompt = ctx.to_messages()
                mem_prompt[0] = {"role": "system", "content": self.system}
                built["memory"] = mem_prompt
                mem_tokens = messages_tokens(mem_prompt)
                meta["memory_budget_tokens"] = ctx.budget_tokens
                meta["memory_used_tokens"] = ctx.used_tokens

                built["matched"] = _truncate_to_budget(list(full_prompt), mem_tokens)

                for cond in self.conditions:
                    prompt = built[cond]
                    examples.append(
                        Example(
                            condition=cond,
                            turn=i,
                            prompt=prompt,
                            completion=completion,
                            prompt_tokens=messages_tokens(prompt),
                            completion_tokens=ctoks,
                            has_prior_assistant=_has_prior_assistant(prompt),
                            meta=dict(meta),
                        )
                    )

            # advance both the raw history and the memory state
            if user:
                history.append({"role": "user", "content": user})
            if assistant:
                history.append({"role": "assistant", "content": assistant})
            if user or assistant:
                engine.process_turn(user, assistant or None)

        return examples

    # -- many conversations -------------------------------------------------
    def export_traces(self, traces: Dict[str, List[dict]]) -> List[Example]:
        out: List[Example] = []
        for tid, turns in traces.items():
            out.extend(self.export_trace(turns, trace_id=tid))
        return out


# ---------------------------------------------------------------------------
# reporting / writing
# ---------------------------------------------------------------------------

def summarize(examples: Sequence[Example]) -> Dict[str, dict]:
    """Per-condition totals: what each training set costs and what it exposes."""
    summary: Dict[str, dict] = {}
    for cond in CONDITIONS:
        subset = [e for e in examples if e.condition == cond]
        if not subset:
            continue
        p = [e.prompt_tokens for e in subset]
        c = [e.completion_tokens for e in subset]
        summary[cond] = {
            "examples": len(subset),
            "prompt_tokens_total": sum(p),
            "prompt_tokens_mean": round(sum(p) / len(p), 1),
            "prompt_tokens_max": max(p),
            "completion_tokens_total": sum(c),
            "sequence_tokens_total": sum(p) + sum(c),
            "share_with_prior_assistant": round(sum(e.has_prior_assistant for e in subset) / len(subset), 3),
        }
    return summary


def render_summary(summary: Dict[str, dict]) -> str:
    base = summary.get("full", {}).get("sequence_tokens_total")
    lines = [
        "| Condition | Examples | Prompt tok (mean) | Prompt tok (max) | Seq tok (total) | vs full | Prompts replaying own output |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        s = summary.get(cond)
        if not s:
            continue
        rel = f"{s['sequence_tokens_total'] / base * 100:.1f}%" if base else "—"
        lines.append(
            f"| {cond} | {s['examples']} | {s['prompt_tokens_mean']:,} | {s['prompt_tokens_max']:,} | "
            f"{s['sequence_tokens_total']:,} | {rel} | {s['share_with_prior_assistant'] * 100:.0f}% |"
        )
    return "\n".join(lines)


def write_jsonl(examples: Sequence[Example], out_dir: str | Path) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for cond in CONDITIONS:
        subset = [e for e in examples if e.condition == cond]
        if not subset:
            continue
        p = out / f"train_{cond}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for e in subset:
                fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        paths[cond] = p
    summary = summarize(examples)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = out / "summary.json"
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", help="chat traces (JSONL / plain text); each is one conversation")
    parser.add_argument("--demo", action="store_true", help="use the built-in demo trace")
    parser.add_argument("--out", required=True, help="output directory for train_<condition>.jsonl + summary.json")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--profile", default="general", help="memory policy profile for the memory condition")
    parser.add_argument("--system", default="neutral",
                        help="'neutral' | 'contract' | literal system prompt text (shared by all conditions)")
    parser.add_argument("--memory-cap", type=int, default=3000, help="active memory cap (tokens)")
    parser.add_argument("--min-turn", type=int, default=1, help="skip examples before this turn index")
    args = parser.parse_args(argv)

    if not args.traces and not args.demo:
        parser.error("give at least one trace or pass --demo")

    system = SYSTEM_CHOICES.get(args.system, args.system)
    exporter = SFTExporter(
        conditions=args.conditions, profile=args.profile, system=system,
        memory_cap_tokens=args.memory_cap, min_turn=args.min_turn,
    )

    traces: Dict[str, List[dict]] = {}
    if args.demo:
        traces["demo"] = demo_trace()
    for p in args.traces:
        traces[Path(p).stem] = load_trace(p)

    examples = exporter.export_traces(traces)
    paths = write_jsonl(examples, args.out)
    summary = summarize(examples)

    print(f"{len(traces)} conversation(s) → {len(examples)} examples "
          f"({len(examples) // max(1, len(args.conditions))} targets × {len(args.conditions)} conditions)\n")
    print(render_summary(summary))
    print("\nwrote:")
    for k, p in paths.items():
        print(f"  {k:<10} {p}")
    print("\nTrain with loss on `completion` only (TRL prompt-completion default). "
          "See docs/training.md for the design and the cross-eval matrix.")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
