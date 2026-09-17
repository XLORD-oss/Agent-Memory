"""Memory-fidelity benchmark tests: state fidelity, dedupe, protection."""

import tempfile

from agent_memory.distiller import RuleDistiller

from benchmarks.common.memory_builder import MarkerDistiller
from benchmarks.fidelity.run import (
    build_engine,
    evaluate,
    filtered_recall,
    replay,
)
from benchmarks.fidelity.tasks import generate_conversation, planted_occurrences


def _run(distiller, cap, facts=24, repeats=2, seed=0, track=True, raw=5000):
    conv = generate_conversation(n_facts=facts, repeats=repeats, seed=seed)
    eng = build_engine(tempfile.mkdtemp(prefix="fid-test-"), distiller, cap, "usage", "general")
    replay(eng, conv, track)
    return eng, conv, evaluate(eng, conv, raw)


def test_generator_shapes():
    conv = generate_conversation(n_facts=10, repeats=3, seed=1)
    assert len(conv.facts) == 10
    assert planted_occurrences(conv) == 30
    assert len({f.value for f in conv.facts}) == 10  # all values distinct
    assert conv.referenced_ids <= {f.id for f in conv.facts}
    assert 0 < len(conv.referenced_ids) < 10
    assert any("[[FACT:" in (t.get("user") or "") for t in conv.turns)


def test_big_cap_holds_everything_and_dedupes():
    eng, conv, row = _run(MarkerDistiller(), cap=5000, facts=10, repeats=3, seed=2)
    assert row.recall_active == 1.0
    assert row.recall_total == 1.0
    # 30 plantings -> 10 distinct facts: merge collapsed the repeats exactly.
    assert row.distinct_facts == 10
    assert row.entries_active == 10


def test_state_fidelity_monotone_with_cap():
    conv = generate_conversation(n_facts=16, repeats=1, seed=3)
    recalls = []
    for cap in (40, 100, 400):
        eng = build_engine(tempfile.mkdtemp(prefix="fid-test-"), MarkerDistiller(), cap, "usage", "general")
        replay(eng, conv, True)
        recalls.append(evaluate(eng, conv, 5000).recall_active)
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[0] < 1.0  # tight cap really does lose active state


def test_archive_is_lossless_audit_trail():
    eng, conv, row = _run(MarkerDistiller(), cap=60, facts=20, repeats=1, seed=4)
    assert row.recall_active < row.recall_total  # active is the lossy view
    assert row.recall_total == 1.0               # archive holds everything
    assert row.lost == 0.0


def test_marker_precision_is_perfect():
    eng, conv, row = _run(MarkerDistiller(), cap=200, facts=10, repeats=1, seed=6)
    assert row.precision == 1.0  # every active entry is a planted fact


def test_usage_protection_referenced_survive_better():
    conv = generate_conversation(n_facts=24, repeats=2, seed=0)
    ref = conv.referenced_ids
    unref = {f.id for f in conv.facts} - ref
    eng = build_engine(tempfile.mkdtemp(prefix="fid-test-"), RuleDistiller(), 160, "usage", "general")
    replay(eng, conv, True)
    ref_recall = filtered_recall(eng, conv, ref)
    unref_recall = filtered_recall(eng, conv, unref)
    # Under usage-weighted eviction, referenced facts hold at least as well as
    # unreferenced ones; with this seed the protection is strongly positive.
    assert ref_recall > unref_recall
    assert ref_recall >= 0.3


def test_usage_tracking_adds_uses_to_referenced_entries():
    conv = generate_conversation(n_facts=10, repeats=1, seed=7)
    eng = build_engine(tempfile.mkdtemp(prefix="fid-test-"), RuleDistiller(), 400, "usage", "general")
    replay(eng, conv, True)
    values = {f.value for f in conv.facts if f.id in conv.referenced_ids}
    referenced_used = [
        e for e in eng.store.all() if any(v in e.text for v in values) and e.uses > 0
    ]
    assert len(referenced_used) >= 1

def test_staleness_metric_supersession_holds_current_value_only():
    import tempfile
    from agent_memory.core import MemoryEngine
    from agent_memory.distiller import RuleDistiller
    from benchmarks.fidelity.run import staleness
    from benchmarks.fidelity.tasks import generate_corrections

    corr = generate_corrections(n=6, seed=3)
    eng = MemoryEngine(state_dir=tempfile.mkdtemp(), distiller=RuleDistiller(), memory_cap_tokens=100_000)
    for t in corr.turns:
        eng.process_turn(t["user"])
    st = staleness(eng, corr.corrections)
    assert st["current"] == 1.0 and st["contradiction"] == 0.0 and st["stale_only"] == 0.0
    assert st["active_entries"] == 6
    # the superseded values are in the archive, not gone
    archived = "".join(p.read_text() for p in eng.store.archive_dir.glob("*.md"))
    assert all(c.old in archived for c in corr.corrections)
