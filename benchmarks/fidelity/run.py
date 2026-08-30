"""Memory-fidelity benchmark — the assimilation error of the compact memory.

The compact memory is a *lossy compression* of the conversation (distillation +
dedupe + archiving). This benchmark measures how much planted ground truth
survives in the memory files themselves — BEFORE any model ever reads them —
isolating the memory from the model. The chaos-researcher reading: this is the
truncation error of a reduced-order model; recall-vs-cap is the rate-distortion
curve of the conversation.

Sources of loss, kept separate:
* **distillation loss** — recall(gap) = MarkerDistiller(perfect) − RuleDistiller(realistic)
* **retention loss**    — recall_active(active store) vs recall_total(active + archive)
* **assimilation error**= 1 − recall_total (what is lost everywhere, i.e. truly gone)

    python -m benchmarks.fidelity.run
    python -m benchmarks.fidelity.run --caps 400,800,1600,3200 --distiller rules
    python -m benchmarks.fidelity.run --plot rate_distortion.png
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory.core import MemoryEngine  # noqa: E402
from agent_memory.distiller import RuleDistiller  # noqa: E402
from agent_memory.policy import TASK_PROFILES  # noqa: E402
from agent_memory.tokens import estimate_tokens  # noqa: E402

from benchmarks.common.harness import write_json  # noqa: E402
from benchmarks.common.memory_builder import MarkerDistiller  # noqa: E402
from benchmarks.fidelity.tasks import Conversation, generate_conversation, planted_occurrences  # noqa: E402


@dataclass
class FidelityRow:
    cap_tokens: int
    recall_active: float
    recall_archive: float
    recall_total: float
    lost: float
    precision: float
    entries_active: int
    entries_total: int
    distinct_facts: int
    memory_tokens: int
    compression: float
    prompt_tokens: int


def build_engine(
    state_dir: str, distiller, cap_tokens: int, archive_policy: str, profile: str
) -> MemoryEngine:
    return MemoryEngine(
        state_dir=state_dir,
        distiller=distiller,
        memory_cap_tokens=cap_tokens,
        archive_policy=archive_policy,
        policy=TASK_PROFILES[profile],
    )


def replay(engine: MemoryEngine, conversation: Conversation, track_usage: bool) -> MemoryEngine:
    for t in conversation.turns:
        # Real agents assemble a prompt every turn; count those tokens too.
        engine.build_context(t.get("user") or "")
        engine.process_turn(
            t.get("user") or "", t.get("assistant"), track_usage=track_usage
        )
    return engine


def evaluate(engine: MemoryEngine, conversation: Conversation, raw_tokens: int) -> FidelityRow:
    facts = conversation.facts
    values = [f.value for f in facts]
    n = len(facts)

    active_entries = engine.store.all()
    active_text = "\n".join(e.text for e in active_entries)
    archive_text = "".join(
        p.read_text(encoding="utf-8") for p in engine.store.archive_dir.glob("*.md")
    )
    all_text = active_text + "\n" + archive_text

    vals_active = sum(1 for v in values if v in active_text)
    vals_archive = sum(1 for v in values if v in archive_text)
    vals_anywhere = sum(1 for v in values if v in all_text)

    # precision: share of active entries that correspond to a planted fact value
    if active_entries:
        hits = sum(1 for e in active_entries if any(v in e.text for v in values))
        precision = hits / len(active_entries)
    else:
        precision = 0.0

    # dedupe: how many DISTINCT facts are represented in the active store
    distinct_facts = sum(1 for v in values if v in active_text)

    archive_entries = [ln for p in engine.store.archive_dir.glob("*.md") for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
    all_texts = [e.text for e in active_entries] + archive_entries

    memory_tokens = engine.memory_tokens()
    return FidelityRow(
        cap_tokens=engine.memory_cap_tokens,
        recall_active=vals_active / n,
        recall_archive=vals_archive / n,
        recall_total=vals_anywhere / n,
        lost=1 - vals_anywhere / n,
        precision=precision,
        entries_active=len(active_entries),
        entries_total=len(all_texts),
        distinct_facts=distinct_facts,
        memory_tokens=memory_tokens,
        compression=raw_tokens / memory_tokens if memory_tokens else 0.0,
        prompt_tokens=engine.total_tokens_processed(),
    )


def filtered_recall(engine: MemoryEngine, conversation: Conversation, ids: set) -> float:
    """Recall over a subset of facts (used for referenced vs unreferenced)."""
    active_text = "\n".join(e.text for e in engine.store.all())
    if not ids:
        return 0.0
    hits = sum(1 for f in conversation.facts if f.id in ids and f.value in active_text)
    return hits / len(ids)


def format_row(row: FidelityRow) -> str:
    return (
        f"| {row.cap_tokens:>5} | {row.recall_active:.0%} | {row.recall_archive:.0%} "
        f"| {row.recall_total:.0%} | {row.lost:.0%} | {row.precision:.0%} "
        f"| {row.entries_active}/{row.entries_total} | {row.distinct_facts} "
        f"| {row.memory_tokens:>4} | {row.compression:.1f}x | {row.prompt_tokens:>7} |"
    )


def print_table(header: str, rows: list) -> None:
    print(f"\n{header}")
    print("| cap | recall active | recall archive | recall total | lost "
          "| precision | entries act/all | distinct facts | mem tok | comp | prompt tok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(format_row(r))


def plot_rate_distortion(rows: list, out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib not installed; pip install 'agent-memory[plot]'") from exc
    xs = [r.cap_tokens for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [r.recall_active for r in rows], "o-", label="Recall in active memory")
    ax.plot(xs, [r.recall_total for r in rows], "s-", label="Recall total (incl. archive)")
    ax.set_xlabel("Memory cap (tokens)")
    ax.set_ylabel("Recall of planted facts")
    ax.set_title("Assimilation: memory size vs fidelity (the conversation's rate-distortion curve)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", type=int, default=24, help="planted ground-truth facts")
    parser.add_argument("--repeats", type=int, default=2, help="plant each fact N times (dedupe test)")
    parser.add_argument("--caps", default="40,80,120,160,240", help="comma list of memory caps (tokens)")
    parser.add_argument("--distiller", choices=["marker", "rules"], default="marker",
                        help="marker=perfect distillation control; rules=realistic RuleDistiller")
    parser.add_argument("--archive-policy", choices=["usage", "oldest"], default="usage")
    parser.add_argument("--no-usages", action="store_true", help="disable usage tracking")
    parser.add_argument("--protection-cap", type=int, default=160,
                        help="cap (tokens) used for the usage-protection comparison")
    parser.add_argument("--profile", default="general", choices=sorted(TASK_PROFILES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="write the rate-distortion PNG here")
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    conversation = generate_conversation(n_facts=args.facts, repeats=args.repeats, seed=args.seed)
    raw_tokens = sum(
        estimate_tokens(t.get("user") or "") + estimate_tokens(t.get("assistant") or "")
        for t in conversation.turns
    )
    planted = planted_occurrences(conversation)

    cap_list = [int(c) for c in args.caps.split(",") if c.strip()]
    print(f"[fidelity] {len(conversation.facts)} facts x{args.repeats} = {planted} planted "
          f"occurrences · {raw_tokens:,} raw tokens · distiller={args.distiller} · "
          f"archive={args.archive_policy} · profile={args.profile}")

    distiller = RuleDistiller() if args.distiller == "rules" else MarkerDistiller()
    track_usage = not args.no_usages

    rows = []
    for cap in cap_list:
        eng = build_engine(
            tempfile.mkdtemp(prefix="fidelity-"), distiller, cap, args.archive_policy, args.profile
        )
        replay(eng, conversation, track_usage)
        rows.append(evaluate(eng, conversation, raw_tokens))

    print_table("STATE FIDELITY (what the model actually sees) — recall in active memory", rows)
    print("  state fidelity = recall_active · archival completeness = recall_total (everything")
    print("  is kept in archive/ as the audit trail; only the ACTIVE memory is lossy by design.")

    # ---- source-of-loss breakdown at the tightest cap ---------------------
    tight_cap = cap_list[0]

    def run_with(d: object, track: bool) -> FidelityRow:
        eng = build_engine(tempfile.mkdtemp(prefix="fidelity-"), d, tight_cap, args.archive_policy, args.profile)
        replay(eng, conversation, track)
        return evaluate(eng, conversation, raw_tokens)

    row_marker = run_with(MarkerDistiller(), True)
    row_rules = run_with(RuleDistiller(), track_usage)
    dist_loss = row_marker.recall_active - row_rules.recall_active
    print(f"\nSource-of-loss breakdown at cap = {tight_cap} (active-memory state fidelity):")
    print(f"  distillation loss (marker − rules state fidelity):  {dist_loss:+.0%} "
          f"(perfect={row_marker.recall_active:.0%}, rules={row_rules.recall_active:.0%})")
    print(f"  staleness (archive covers active gaps, marker):     "
          f"{max(0.0, row_marker.recall_total - row_marker.recall_active):.0%} "
          f"(recall_total={row_marker.recall_total:.0%})")

    # ---- usage-weighted protection (realistic distiller only) -------------
    if not args.no_usages:
        ref = conversation.referenced_ids
        unref = {f.id for f in conversation.facts} - ref
        prot_cap = max(tight_cap, args.protection_cap)
        eng_ref = build_engine(tempfile.mkdtemp(prefix="fidelity-"), RuleDistiller(), prot_cap, args.archive_policy, args.profile)
        replay(eng_ref, conversation, True)
        ref_active = filtered_recall(eng_ref, conversation, ref)
        unref_active = filtered_recall(eng_ref, conversation, unref)
        eng_off = build_engine(tempfile.mkdtemp(prefix="fidelity-"), RuleDistiller(), prot_cap, args.archive_policy, args.profile)
        replay(eng_off, conversation, False)
        off_active = filtered_recall(eng_off, conversation, ref)
        print(f"\nUsage-weighted protection at cap={prot_cap} (rules distiller):")
        print(f"  state fidelity — referenced facts: {ref_active:.0%} vs unreferenced: {unref_active:.0%} "
              f"({ref_active - unref_active:+.0%} protection)")
        print(f"  vs no usage tracking: referenced state fidelity {off_active:.0%} "
              f"({ref_active - off_active:+.0%} gain)")

    # ---- dedupe check -----------------------------------------------------
    print(f"\nDedupe: {planted} planted occurrences -> {rows[-1].distinct_facts} distinct facts "
          f"in active memory at the largest cap (ideal = {len(conversation.facts)}) "
          f"— repeats collapsed by merge.")

    if args.plot:
        plot_rate_distortion(rows, args.plot)

    if args.out:
        write_json(
            args.out,
            {
                "facts": len(conversation.facts),
                "repeats": args.repeats,
                "planted_occurrences": planted,
                "raw_tokens": raw_tokens,
                "distiller": args.distiller,
                "archive_policy": args.archive_policy,
                "rows": [r.__dict__ for r in rows],
                "distillation_loss": dist_loss,
            },
        )


if __name__ == "__main__":
    main()