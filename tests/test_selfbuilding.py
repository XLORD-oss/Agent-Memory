"""The three gaps that made the map demo-built rather than self-building.

G1 — edges are traversed at selection time (an argument brings its premises).
G2 — the rich kinds (principle / profile / argument / perspective) are extracted
     automatically by both distillers, and premises are linked on merge.
G3 — a correction supersedes its predecessor instead of adding a contradiction.
"""

from __future__ import annotations

import glob
import json
import tempfile

import pytest

from agent_memory import MemoryEngine, MemoryPolicy
from agent_memory.distiller import LLMDistiller, RuleDistiller, Update, extraction_from_json
from agent_memory.policy import expand_selection, score_item
from agent_memory.storage import MemoryEntry, subject_overlap


class Fixture:
    """A client that replays a canned distiller JSON payload."""

    def __init__(self, payload):
        self.payload = payload

    def complete(self, messages, **kw):
        return json.dumps(self.payload)


def _engine(**kw) -> MemoryEngine:
    return MemoryEngine(state_dir=tempfile.mkdtemp(), **kw)


def _archive_lines(engine: MemoryEngine):
    text = "".join(open(f).read() for f in glob.glob(str(engine.store.archive_dir / "*.md")))
    return [l for l in text.splitlines() if l.startswith("- ")]


# ---------------------------------------------------------------------------
# G3 — supersession
# ---------------------------------------------------------------------------

def test_subject_overlap_sees_through_value_changes():
    assert subject_overlap("the demo is on Friday", "the demo is on Monday") >= 0.5
    assert subject_overlap("I prefer meetings after 10am", "I prefer meetings after 11am") >= 0.5
    # a refinement about a different subject does not look like a correction
    assert subject_overlap("build the prototype with FastAPI and Postgres", "use SQLite for tests, not Postgres") < 0.5


def test_correction_retires_predecessor_and_archives_with_pointer():
    e = _engine()
    e.process_turn("Remember that the demo is on Friday.")
    e.process_turn("Actually, the demo is on Monday.")
    facts = e.store.all(kind="fact")
    assert [f.text for f in facts] == ["the demo is on Monday"]
    assert len(facts[0].supersedes) == 1
    lines = _archive_lines(e)
    assert len(lines) == 1 and "Friday" in lines[0] and "superseded by" in lines[0]
    # the retired statement never re-enters the MEMORY section (the recent
    # user-turn window may still echo the raw sentence — that is flow context)
    assert "Friday" not in e.build_context("when is the demo?").memory
    assert "Monday" in e.build_context("when is the demo?").memory


def test_refinement_does_not_supersede():
    e = _engine()
    e.process_turn("We decided to build the prototype with FastAPI and Postgres.")
    e.process_turn("Actually, use SQLite for tests, not Postgres.")
    texts = sorted(c.text for c in e.store.all(kind="conclusion"))
    assert len(texts) == 2 and any("FastAPI" in t for t in texts) and any("SQLite" in t for t in texts)
    assert _archive_lines(e) == []


def test_restatement_neither_duplicates_nor_retires():
    e = _engine()
    e.process_turn("Remember that the demo is on Friday.")
    e.process_turn("Actually, the demo is on Friday.")
    assert [f.text for f in e.store.all(kind="fact")] == ["the demo is on Friday"]
    assert _archive_lines(e) == []


def test_successor_inherits_usage_pin_links_and_inbound_edges():
    e = _engine()
    old = e.add_entry("the demo is on Friday", kind="fact", priority=2.0, pinned=True)
    prem = e.add_entry("the funder visits that day", kind="fact")
    e.link(old.id, prem.id)
    arg = e.add_argument("ship before the demo", premises=[old.id])
    old.mark_used(); old.mark_used()
    new = e.supersede(old.id, "the demo is on Monday")
    assert new.kind == "fact" and new.uses == 2 and new.pinned and new.priority == 2.0
    assert prem.id in new.links and old.id in new.supersedes
    assert e.store.get(arg.id).links == [new.id]  # inbound edge redirected
    assert e.store.get(old.id) is None


def test_llm_update_with_old_hint_and_kind_preserved():
    e = _engine(distiller=LLMDistiller(Fixture({"preferences": ["I prefer plots in matplotlib"]})))
    e.process_turn("x", "y")
    e.distiller = LLMDistiller(Fixture({"updates": [{"old": "prefer plots in matplotlib",
                                                     "new": "I prefer plots in plotly now", "kind": "preference"}]}))
    e.process_turn("x", "y")
    prefs = e.store.all(kind="preference")
    assert [p.text for p in prefs] == ["I prefer plots in plotly now"]


def test_arguments_and_perspectives_are_never_superseded_by_corrections():
    e = _engine()
    arg = e.add_argument("the ensemble mean beats the control run")
    e.apply_update(Update(old_hint="ensemble mean beats control", new=MemoryEntry(text="the ensemble mean beats the control run only sometimes", kind="fact")))
    assert e.store.get(arg.id) is not None


# ---------------------------------------------------------------------------
# G2 — rich extraction
# ---------------------------------------------------------------------------

def test_rule_distiller_extracts_profile_principle_perspectives():
    d = RuleDistiller()
    ext = d.distill("I am a researcher in nonlinear dynamics and I work on monsoon predictability.", "", 1)
    kinds = {(x.kind, x.tags[0]) for x in ext.profile}
    assert ("profile", "identity") in kinds and ("profile", "domain") in kinds
    assert any(x.text == "researcher in nonlinear dynamics" for x in ext.profile)

    ext = d.distill("Principle: forecast skill is bounded by initial-condition uncertainty.", "", 1)
    assert [p.text for p in ext.principles] == ["forecast skill is bounded by initial-condition uncertainty"]

    ext = d.distill("On one hand Postgres handles the joins; on the other hand it is an ops burden.", "", 1)
    assert {(p.stance, p.text) for p in ext.perspectives} == {("for", "Postgres handles the joins"), ("against", "it is an ops burden")}


def test_rule_distiller_correction_is_not_also_a_fresh_fact():
    ext = RuleDistiller().distill("Actually, the demo is on Monday.", "", 1)
    assert len(ext.updates) == 1 and ext.updates[0].new.text == "the demo is on Monday"
    assert ext.facts == [] and ext.conclusions == []


def test_rule_distiller_sentence_boundaries():
    ext = RuleDistiller().distill("We decided to use FastAPI. Remember that the demo is Friday.", "", 1)
    assert [c.text for c in ext.conclusions] == ["use FastAPI"]
    assert [f.text for f in ext.facts] == ["the demo is Friday"]


def test_llm_schema_covers_the_whole_map_and_links_premises():
    payload = {
        "facts": ["The Lyapunov spectrum is computed with the Benettin method."],
        "principles": ["Forecast skill is bounded by initial-condition uncertainty."],
        "profile": [{"field": "identity", "text": "researcher in nonlinear dynamics"},
                    {"field": "bogus", "text": "falls back to identity"}],
        "arguments": [{"claim": "The ensemble mean beats the control run.",
                       "premises": ["Ensemble averaging filters the unstable directions.",
                                    "The control run sits on one trajectory."]}],
        "perspectives": [{"question": "Use Postgres?", "stance": "for", "text": "handles the joins"},
                         {"question": "Use Postgres?", "stance": "weird", "text": "unclear"}],
        "updates": [], "nothing_new": False,
    }
    ext = extraction_from_json(payload, turn_number=3)
    assert len(ext.arguments) == 1 and len(ext.premises_of[ext.arguments[0].id]) == 2
    assert {p.tags[0] for p in ext.profile} == {"identity"}
    assert {p.stance for p in ext.perspectives} == {"for", "open"}
    assert len(ext) == 1 + 1 + 2 + 1 + 2 + 2  # fact, principle, profile x2, argument, premises x2, perspectives x2

    e = _engine(distiller=LLMDistiller(Fixture(payload)))
    n = e.process_turn("research message", "ok")
    assert n == 9
    arg = e.store.all(kind="argument")[0]
    assert len(arg.links) == 2 and all(e.store.get(l) is not None for l in arg.links)
    ctx = e.build_context("should I use postgres?")
    for section in ("profile", "principles", "memory", "perspectives", "arguments"):
        assert getattr(ctx, section).strip(), section
    assert "← Ensemble averaging filters the unstable directions" in ctx.arguments
    assert "**FOR** handles the joins" in ctx.perspectives


def test_llm_premise_dedupes_against_existing_memory():
    e = _engine()
    existing = e.add_entry("Ensemble averaging filters the unstable directions", kind="fact")
    e.distiller = LLMDistiller(Fixture({"arguments": [{"claim": "The ensemble mean beats the control run",
                                                       "premises": ["Ensemble averaging filters the unstable directions"]}]}))
    e.process_turn("x", "y")
    arg = e.store.all(kind="argument")[0]
    assert arg.links == [existing.id]
    assert len(e.store.all(kind="fact")) == 1


def test_llm_distiller_tolerates_garbage():
    e = _engine(distiller=LLMDistiller(Fixture("not json at all")))
    assert e.process_turn("x", "y") == 0
    ext = extraction_from_json({"facts": [None, "", {"text": "ok"}]}, 1)
    assert [f.text for f in ext.facts] == ["ok"]


# ---------------------------------------------------------------------------
# G1 — edges at selection time
# ---------------------------------------------------------------------------

def _graph_engine():
    e = _engine()
    p1 = e.add_entry("Close trajectories diverge exponentially in a chaotic system.", kind="fact", priority=0.2)
    p2 = e.add_entry("Initial conditions are only known to finite precision.", kind="fact", priority=0.2)
    for i in range(12):
        e.add_entry(f"Distractor fact number {i} about something unrelated to dynamics.", kind="fact", priority=1.0)
    arg = e.add_argument("Forecast skill is bounded no matter how good the model is.", premises=[p1.id, p2.id])
    e.set_priority(arg.id, 3.0)
    return e, arg, p1, p2


def test_inbound_links_raise_score():
    pol = MemoryPolicy()
    lone = score_item("x", "fact", 1.0, 0, 0, pol, inbound_links=0)
    premise = score_item("x", "fact", 1.0, 0, 0, pol, inbound_links=2)
    assert premise > lone
    flat = MemoryPolicy(link_weight=0.0)
    assert score_item("x", "fact", 1.0, 0, 0, flat, inbound_links=2) == score_item("x", "fact", 1.0, 0, 0, flat)


@pytest.mark.parametrize("budget", [60, 120])
def test_selected_argument_brings_its_premises_within_budget(budget):
    e, arg, p1, p2 = _graph_engine()
    flat = e.build_context("q", policy=MemoryPolicy(budget_tokens=budget, expand_links=False, link_weight=0.0))
    mapped = e.build_context("q", policy=MemoryPolicy(budget_tokens=budget))
    assert arg.id in flat.linked_in and not (p1.id in flat.linked_in and p2.id in flat.linked_in)
    assert arg.id in mapped.linked_in and p1.id in mapped.linked_in and p2.id in mapped.linked_in
    assert mapped.used_tokens <= budget
    assert "← Close trajectories diverge" in mapped.arguments


def test_expand_selection_drops_unsupportable_claim_rather_than_showing_it_bare():
    cands = [
        {"id": "arg", "kind": "argument", "score": 5.0, "tokens": 10, "links": ["p"]},
        {"id": "p", "kind": "fact", "score": 1.0, "tokens": 50, "links": []},
        {"id": "pin", "kind": "principle", "score": 9.0, "tokens": 10, "links": []},
    ]
    sel = [cands[0], cands[2]]
    out = expand_selection(sel, cands, budget_tokens=30)
    ids = {c["id"] for c in out}
    assert "pin" in ids and "arg" not in ids and "p" not in ids  # cannot fit the premise -> claim dropped, not shown bare


def test_expand_never_admits_zero_score_dependencies():
    cands = [
        {"id": "arg", "kind": "argument", "score": 5.0, "tokens": 10, "links": ["a"]},
        {"id": "a", "kind": "assistant_turn", "score": 0.0, "tokens": 5, "links": []},
    ]
    out = expand_selection([cands[0]], cands, budget_tokens=100)
    assert [c["id"] for c in out] == ["arg"]


def test_context_linked_in_lists_prompt_ids():
    e, arg, p1, p2 = _graph_engine()
    ctx = e.build_context("q", policy=MemoryPolicy(budget_tokens=5000))
    assert set(ctx.linked_in) == {x.id for x in e.store.all()}
