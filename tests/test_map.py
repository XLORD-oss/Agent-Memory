"""Tests for the memory-as-map structure: new kinds, edges, principles,
perspectives, arguments, and the user profile."""

import tempfile

from agent_memory.core import MemoryEngine, _pick_victim
from agent_memory.storage import MemoryEntry, MemoryStore, PIN_PRIORITY, PROTECTED_KINDS


def _engine(**kw):
    defaults = dict(state_dir=tempfile.mkdtemp())
    defaults.update(kw)
    return MemoryEngine(**defaults)


# --- the new entry kinds -----------------------------------------------------

def test_add_entry_supports_all_kinds():
    eng = _engine()
    for kind in ("fact", "conclusion", "preference", "principle", "argument", "perspective", "profile"):
        e = eng.add_entry(text=f"a {kind}", kind=kind)
        assert e.kind == kind
        assert eng.store.get(e.id) is not None


def test_add_principle():
    eng = _engine()
    p = eng.add_principle("Sensitivity to initial conditions bounds forecast horizon.", domain="chaos")
    assert p.kind == "principle"
    assert p.priority >= 1.0
    assert "Sensitivity" in eng.principles_md


def test_add_profile_entry():
    eng = _engine()
    eng.add_profile_entry("identity", "I am a researcher working on chaos applied to weather.")
    eng.add_profile_entry("style", "I want compact, quantitative answers.")
    ctx = eng.build_context("who am I?")
    assert "USER PROFILE" in ctx.user_prompt
    assert "researcher" in ctx.user_prompt
    assert "Identity" in ctx.profile or "identity" in ctx.profile.lower()


def test_add_profile_entry_rejects_bad_field():
    eng = _engine()
    try:
        eng.add_profile_entry("hobby", "stamps")
        assert False, "should have raised"
    except ValueError:
        pass


def test_add_argument_with_premises():
    eng = _engine()
    p1 = eng.add_argument("Premise: initial error grows as e^(lambda t).", premises=[])
    p2 = eng.add_argument("Premise: lambda > 0 for the Lorenz attractor.", premises=[])
    claim = eng.add_argument(
        "Therefore forecast skill is inherently bounded.",
        premises=[p1.id, p2.id],
    )
    assert claim.kind == "argument"
    assert claim.links == [p1.id, p2.id]
    assert eng.related(claim.id) == [p1, p2]  # traversable map


def test_add_perspective_holds_conflicting_views():
    eng = _engine()
    for_stance = eng.add_perspective("use Postgres?", "for", "it handles the ensemble data well.")
    against = eng.add_perspective("use Postgres?", "against", "it adds ops burden for this scale.")
    eng.add_perspective("use Postgres?", "open", "wait until the benchmark.")

    ctx = eng.build_context("what's the database decision?")
    assert "PERSPECTIVES" in ctx.user_prompt
    assert "FOR" in ctx.user_prompt and "AGAINST" in ctx.user_prompt
    assert len(eng.store.all("perspective")) == 3  # all coexisting


def test_link_and_related():
    eng = _engine()
    a = eng.add_entry("the ENSO teleconnection", kind="fact", domain="climate")
    b = eng.add_entry("monsoon onset correlates with ENSO", kind="fact", domain="climate")
    eng.link(a.id, b.id)
    assert eng.related(a.id) == [b]
    assert eng.related(b.id) == []  # directed edge


def test_principles_never_evicted():
    eng = _engine(memory_cap_tokens=60)
    eng.add_principle("Forecasts degrade with initial-condition uncertainty.")
    # Flood memory well past the cap.
    for i in range(30):
        eng.process_turn(f"Remember that the run number is {i} and it matters a lot.", "ok")
    eng.archive_if_needed(policy="usage")

    assert eng.memory_tokens() <= 60
    assert len(eng.store.all("principle")) == 1  # the axiom survived


def test_protected_kinds_include_principles():
    assert "principle" in PROTECTED_KINDS
    assert "conclusion" in PROTECTED_KINDS


def test_pick_victim_never_protected():
    eng = _engine(memory_cap_tokens=60)
    eng.add_principle("the axiom")
    vuln = eng.add_entry("disposable fact", kind="fact")
    victim = _pick_victim(eng.store.all(), engine=eng, archive_policy="usage")
    assert victim.id == vuln.id  # protected principle never picked


def test_render_markdown_backlinks_maps():
    eng = _engine()
    a = eng.add_entry("the ENSO index", kind="fact")
    b = eng.add_entry("monsoon phase", kind="fact")
    eng.link(b.id, a.id)
    md = eng.memory_md
    assert "monsoon phase" in md
    assert "«" in md  # backlink rendered
    assert "ENSO" in md


# --- persistence / backward compat ------------------------------------------

def test_links_and_new_fields_persist():
    eng = _engine()
    a = eng.add_entry("fact a", kind="fact")
    b = eng.add_entry("fact b", kind="fact")
    eng.link(a.id, b.id)
    eng.add_perspective("q?", "for", "because x", links=[b.id])

    eng2 = MemoryEngine(state_dir=eng.store.root)
    a2 = eng2.store.get(a.id)
    assert a2.links == [b.id]
    pers = eng2.store.all("perspective")
    assert len(pers) == 1
    assert pers[0].stance == "for"
    assert pers[0].links == [b.id]


def test_v3_state_without_new_fields_loads():
    store = MemoryStore(tempfile.mkdtemp())
    store.add(MemoryEntry(text="old fact", kind="fact"))
    store.save()
    # Simulate a v3 file: strip the new fields.
    import json
    data = json.loads(store.state_path.read_text())
    data["version"] = 3
    for e in data["entries"]:
        e.pop("links", None)
        e.pop("stance", None)
        e.pop("domain", None)
    store.state_path.write_text(json.dumps(data))

    store2 = MemoryStore(store.root)
    assert len(store2.all()) == 1
    entry = store2.all()[0]
    assert entry.links == []
    assert entry.stance == ""
    assert entry.domain == ""
    assert entry.text == "old fact"


def test_pin_priority_constant():
    eng = _engine()
    e = eng.add_entry("x", kind="fact")
    eng.pin(e.id)
    assert eng.store.all("fact")[0].effective_priority() == PIN_PRIORITY


def test_principles_and_profile_written_to_disk():
    eng = _engine()
    eng.add_principle("first principle about predictability")
    eng.add_profile_entry("domain", "chaos applied to weather")
    eng.store.write_all_md()
    assert (eng.store.root / "principles.md").exists()
    assert (eng.store.root / "profile.md").exists()
    assert "predictability" in (eng.store.root / "principles.md").read_text()
    assert "weather" in (eng.store.root / "profile.md").read_text()