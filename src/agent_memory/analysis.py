"""Trace analysis: replay real chat logs through the engine and quantify.

Produces, per policy:
* per-turn token series — naive full-history replay vs compact memory
* totals, ratio, and % saved
* memory behavior — entries created / archived / survived, usage report,
  compression ratio, evicted list
* an optional per-policy comparison matrix and (with matplotlib) a plot

Formats (auto-detected):
* JSONL, one JSON object per line
  - ``{"role": "user"|"assistant", "content": "..."}``  (OpenAI-style)
  - ``{"user": "...", "assistant": "..."}``              (paired turns)
* plain text, one turn per line prefixed ``user:`` / ``assistant:``

CLI:
    python -m agent_memory.analysis --trace trace.jsonl --profile coding
    python -m agent_memory.analysis --trace trace.jsonl --compare --plot cost.png
    python -m agent_memory.analysis --trace demo                 # built-in sample

The ``LLMDistiller`` is the production distillation path; without it the trace
is distilled by the deterministic ``RuleDistiller`` (so distillation yield on
arbitrary logs is reported honestly, not faked).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .context import SYSTEM_PROMPT
from .core import MemoryEngine
from .distiller import DistillerLike
from .policy import TASK_PROFILES, MemoryPolicy
from .storage import MemoryStore
from .tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace(path: str | Path) -> List[dict]:
    """Load a chat trace into a list of paired ``{"user":..., "assistant":...}`` turns."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".json") or text.lstrip().startswith("{"):
        return _load_json_trace(text)
    return _load_plain_trace(text)


def _load_json_trace(text: str) -> List[dict]:
    pairs: List[dict] = []
    pending_user: Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "role" in obj:  # OpenAI-style message log
            role = str(obj.get("role", "")).lower()
            content = str(obj.get("content") or obj.get("text") or "").strip()
            if not content:
                continue
            if role == "user":
                if pending_user is not None:
                    pairs.append({"user": pending_user, "assistant": None})
                pending_user = content
            elif role in ("assistant", "model", "ai"):
                pairs.append({"user": pending_user or "", "assistant": content})
                pending_user = None
            elif role == "system":
                continue
        elif "user" in obj or "assistant" in obj:
            pairs.append(
                {
                    "user": str(obj.get("user") or "").strip() or None,
                    "assistant": str(obj.get("assistant") or "").strip() or None,
                }
            )
    if pending_user is not None:
        pairs.append({"user": pending_user, "assistant": None})
    return [p for p in pairs if p.get("user") or p.get("assistant")]


def _load_plain_trace(text: str) -> List[dict]:
    pairs: List[dict] = []
    pending_user: Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("user:") or low.startswith("u:"):
            if pending_user is not None:
                pairs.append({"user": pending_user, "assistant": None})
            pending_user = line.split(":", 1)[1].strip()
        elif low.startswith("assistant:") or low.startswith("a:"):
            pairs.append({"user": pending_user or "", "assistant": line.split(":", 1)[1].strip()})
            pending_user = None
    if pending_user is not None:
        pairs.append({"user": pending_user, "assistant": None})
    return [p for p in pairs if p.get("user") or p.get("assistant")]


def demo_trace() -> List[dict]:
    """A realistic offline sample: a research/coding-style session with facts."""
    return [
        {"user": "Remember that I work at NASA researching chaos theory, mostly driven nonlinear systems.", "assistant": "Noted — driven nonlinear systems, NASA."},
        {"user": "We decided to build the prototype in Python with a FastAPI backend and Postgres.", "assistant": "Python + FastAPI + Postgres, got it."},
        {"user": "I prefer concise answers with bullet points, and never schedule meetings before 10am.", "assistant": "Concise, bullet points, no early meetings."},
        {"user": "The API returns 500 for /users; check the query builder.", "assistant": "```python\ndef get_user(id):\n    user = db.query(User).filter(User.id == id)\n    return user\n```\nThe ORM call looks fine to me — let me check the connection pool."},
        {"user": "fix the TypeError in the transform — it's the same bug from the batch job.", "assistant": "```python\ndef transform(row):\n    return {k: float(v) for k, v in row.items() if v is not None}\n```\nThis mirrors the batch fix we did last sprint."},
        {"user": "The demo for the funder is the last Friday of the month, keep Postgres and FastAPI in the stack for it.", "assistant": "Locked: demo on the last Friday, stack stays as-is."},
        {"user": "Actually, use SQLite for tests, not Postgres.", "assistant": "SQLite for tests, Postgres in prod."},
        {"user": "What's the model architecture for the Lyapunov spectrum?", "assistant": "We use the observed Jacobian estimator with a delay embedding — standard for driven systems."},
        {"user": "can you write the test for the transform?", "assistant": "```python\ndef test_transform():\n    assert transform({'a': '1.5'}) == {'a': 1.5}\n```"},
        {"user": "The Lyapunov spectrum needs the time-delay parameter; derive it from the autocorrelation.", "assistant": "Set delay = first zero of the autocorrelation — that keeps the embedding faithful."},
        {"user": "how do I get the docs to build?", "assistant": "```bash\nmake docs\n```\nNeeds mkdocs + the API stubs regenerated first."},
        {"user": "We concluded that archiving old facts keeps the active context small.", "assistant": "Agreed — that's the retention policy."},
    ]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    turns: int
    raw_tokens: int
    naive_total: int
    compact_total: int
    max_naive_turn: int
    max_compact_turn: int
    entries_created: int
    entries_archived: int
    entries_surviving: int
    memory_tokens_end: int
    compression_ratio: float
    usage: List[dict] = field(default_factory=list)
    evicted: List[str] = field(default_factory=list)
    series: List[dict] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.compact_total / self.naive_total if self.naive_total else float("inf")

    @property
    def saved_pct(self) -> float:
        return (1 - self.ratio) * 100

    def __post_init__(self) -> None:
        if self.memory_tokens_end:
            self.compression_ratio = self.raw_tokens / self.memory_tokens_end


class TraceAnalyzer:
    """Replay a trace through a fresh engine under a chosen policy and measure."""

    def __init__(
        self,
        profile: str = "general",
        policy: Optional[MemoryPolicy] = None,
        distiller: Optional[DistillerLike] = None,
        memory_cap_tokens: int = 3000,
        recent_window: int = 4,
    ) -> None:
        self.profile = profile
        self.policy = policy
        self.distiller = distiller
        self.memory_cap_tokens = memory_cap_tokens
        self.recent_window = recent_window
        self.engine: Optional[MemoryEngine] = None

    def _fresh_engine(self, state_dir: str) -> MemoryEngine:
        return MemoryEngine(
            state_dir=state_dir,
            distiller=self.distiller,
            memory_cap_tokens=self.memory_cap_tokens,
            recent_window=self.recent_window,
            policy=self.policy,
        )

    def analyze(self, turns: List[dict], state_dir: Optional[str] = None) -> AnalysisResult:
        import tempfile

        eng = self._fresh_engine(state_dir or tempfile.mkdtemp(prefix="trace-"))
        self.engine = eng

        sys_tokens = estimate_tokens(SYSTEM_PROMPT)
        raw_tokens = 0
        naive_total = 0
        compact_total = 0
        max_naive_turn = 0
        max_compact_turn = 0
        series: List[dict] = []

        for i, turn in enumerate(turns, start=1):
            user = turn.get("user") or ""
            assistant = turn.get("assistant") or ""

            # Naive replay: the model sees the entire raw conversation so far.
            raw_tokens += estimate_tokens(user) + estimate_tokens(assistant)
            naive_turn = sys_tokens + raw_tokens
            naive_total += naive_turn
            max_naive_turn = max(max_naive_turn, naive_turn)

            # Compact: memory distilled from previous turns + bounded flow/working memory.
            ctx = eng.build_context(user, profile=self.profile, policy=self.policy)
            compact_turn = ctx.prompt_tokens
            compact_total += compact_turn
            max_compact_turn = max(max_compact_turn, compact_turn)

            # Ingest this turn (distill, merge, usage-track, archive).
            if assistant:
                eng.process_turn(user, assistant, track_usage=True)
            else:
                eng.process_turn(user, None, track_usage=False)

            series.append({"turn": i, "naive": naive_turn, "compact": compact_turn})

        surviving = eng.store.all()

        result = AnalysisResult(
            turns=len(turns),
            raw_tokens=raw_tokens,
            naive_total=naive_total,
            compact_total=compact_total,
            max_naive_turn=max_naive_turn,
            max_compact_turn=max_compact_turn,
            entries_created=eng.entry_count,
            entries_archived=_count_archived(eng),  # archived during process_turn
            entries_surviving=len(surviving),
            memory_tokens_end=eng.memory_tokens(),
            compression_ratio=0.0,
            usage=eng.usage_report(top=20),
            evicted=[],
            series=series,
        )
        result.evicted = _archived_entries(eng)
        result.compression_ratio = raw_tokens / eng.memory_tokens() if eng.memory_tokens() else 0.0
        return result

    def render_markdown(self, result: AnalysisResult) -> str:
        lines = [
            f"## Trace analysis — profile `{self.profile}`",
            "",
            f"- turns: **{result.turns}** · raw transcript: **{result.raw_tokens:,}** tokens",
            f"- naive full-history replay: **{result.naive_total:,}** tokens total "
            f"(last turn {result.max_naive_turn:,})",
            f"- compact memory: **{result.compact_total:,}** tokens total "
            f"(last turn {result.max_compact_turn:,})",
            f"- **ratio {result.ratio:.3f} → {result.saved_pct:.1f}% saved**",
            f"- memory at end: **{result.memory_tokens_end:,}** tokens "
            f"({result.compression_ratio:.0f}x smaller than the raw transcript)",
            f"- entries: {result.entries_created} created, {result.entries_archived} archived, "
            f"{result.entries_surviving} surviving",
            "",
            "### Per-turn tokens",
            "",
            "| Turn | Naive replay | Compact | Savings |",
            "|---|---|---|---|",
        ]
        for s in result.series:
            saved = (1 - s["compact"] / s["naive"]) * 100 if s["naive"] else 0
            lines.append(f"| {s['turn']} | {s['naive']:,} | {s['compact']:,} | {saved:.0f}% |")

        if result.usage:
            lines += ["", "### Usage report (what the model actually references)", "",
                      "| uses | kind | entry |", "|---|---|---|"]
            for u in result.usage[:10]:
                lines.append(f"| {u['uses']} | {u['kind']} | {u['text'][:70]} |")

        if result.evicted:
            lines += ["", "### Archived (evicted from active memory)", ""]
            for e in result.evicted[:10]:
                lines.append(f"- [{e['kind']}] {e['text'][:70]}")
        return "\n".join(lines)

    def plot(self, result: AnalysisResult, out_path: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("matplotlib not installed; pip install 'agent-memory[plot]'") from exc
        xs = [s["turn"] for s in result.series]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, [s["naive"] for s in result.series], "o-", label="Naive full-history replay")
        ax.plot(xs, [s["compact"] for s in result.series], "s-", label=f"Compact memory ({self.profile})")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Context tokens")
        ax.set_title(f"Context size per turn — {self.profile} profile")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"wrote {out_path}")


def _count_archived(eng: MemoryEngine) -> int:
    n = 0
    for p in eng.store.archive_dir.glob("*.md"):
        n += p.read_text().count("- ")
    return n


def _archived_entries(eng: MemoryEngine) -> List[dict]:
    out: List[dict] = []
    for p in sorted(eng.store.archive_dir.glob("*.md")):
        for line in p.read_text().splitlines():
            if line.startswith("- "):
                body = line[2:]
                kind = "fact"
                for k, label in (("conclusion", "◆"), ("preference", "★"), ("fact", "•")):
                    if label in body and "[" + k + "]" in body:
                        kind = k
                out.append({"kind": kind, "text": body})
    return out


def compare_policies(
    turns: List[dict],
    profiles: Optional[List[str]] = None,
    distiller: Optional[DistillerLike] = None,
    memory_cap_tokens: int = 3000,
) -> Dict[str, dict]:
    """Run the trace through several profiles and summarize each."""
    profiles = profiles or list(TASK_PROFILES)
    out: Dict[str, dict] = {}
    for profile in profiles:
        a = TraceAnalyzer(profile=profile, distiller=distiller, memory_cap_tokens=memory_cap_tokens)
        r = a.analyze(turns)
        out[profile] = {
            "naive_total": r.naive_total,
            "compact_total": r.compact_total,
            "ratio": r.ratio,
            "saved_pct": r.saved_pct,
            "max_compact_turn": r.max_compact_turn,
            "entries_created": r.entries_created,
            "entries_archived": r.entries_archived,
        }
    return out


def compare_table(results: Dict[str, dict]) -> str:
    lines = [
        "| Profile | Compact total | Naive total | Ratio | Saved | Max ctx/turn | Entries (kept/archived) |",
        "|---|---|---|---|---|---|---|",
    ]
    for profile, r in sorted(results.items(), key=lambda kv: kv[1]["ratio"]):
        lines.append(
            f"| {profile} | {r['compact_total']:,} | {r['naive_total']:,} | {r['ratio']:.3f} | "
            f"{r['saved_pct']:.1f}% | {r['max_compact_turn']:,} | {r['entries_created']}/{r['entries_archived']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a chat trace through compact memory and quantify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trace", default="demo", help="path to trace (jsonl/json/txt), or 'demo'")
    parser.add_argument("--repeat", type=int, default=1, help="repeat the trace N times (longer session)")
    parser.add_argument("--profile", default="general", help="general|coding|research|writing")
    parser.add_argument("--compare", action="store_true", help="run all profiles and compare")
    parser.add_argument("--memory-cap", type=int, default=3000, help="memory token cap")
    parser.add_argument("--plot", default=None, help="write a matplotlib chart to this path")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    turns = load_trace(args.trace) if args.trace != "demo" else demo_trace()
    if args.repeat > 1:
        turns = turns * args.repeat
    print(f"[trace] {len(turns)} turns, {sum(estimate_tokens(t.get('user') or '') + estimate_tokens(t.get('assistant') or '') for t in turns):,} raw tokens\n")

    if args.compare:
        results = compare_policies(turns, memory_cap_tokens=args.memory_cap)
        print(compare_table(results))
        if args.plot:
            import tempfile

            for profile in results:
                a = TraceAnalyzer(profile=profile, memory_cap_tokens=args.memory_cap)
                r = a.analyze(turns)
                a.plot(r, args.plot.replace(".png", f"_{profile}.png"))
    else:
        a = TraceAnalyzer(profile=args.profile, memory_cap_tokens=args.memory_cap)
        r = a.analyze(turns)
        md = a.render_markdown(r)
        print(md)
        if args.plot:
            a.plot(r, args.plot)

    if args.out:
        results = compare_policies(turns, memory_cap_tokens=args.memory_cap) if args.compare else {args.profile: {
            "naive_total": r.naive_total, "compact_total": r.compact_total, "ratio": r.ratio,
            "saved_pct": r.saved_pct, "max_compact_turn": r.max_compact_turn,
        }}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()