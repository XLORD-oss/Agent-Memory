"""Rule-distiller tests: conservative extraction of durable statements."""

from agent_memory.distiller import RuleDistiller
from agent_memory.storage import MemoryEntry


def test_extracts_remember_fact():
    ext = RuleDistiller().distill(
        "Hey! Remember that the deploy happens every Friday at 9am.", "", 1
    )
    assert len(ext.facts) >= 1
    assert "deploy" in ext.facts[0].text.lower()


def test_extracts_preferences():
    ext = RuleDistiller().distill("I prefer concise answers with bullet points.", "", 2)
    kinds = {e.kind for e in ext.all()}
    assert "preference" in kinds
    pref = [e for e in ext.preferences if "concise" in e.text.lower()]
    assert pref


def test_extracts_conclusions():
    ext = RuleDistiller().distill("We decided to migrate to Postgres next quarter.", "", 3)
    assert any("Postgres" in e.text for e in ext.conclusions)


def test_ignores_small_talk():
    ext = RuleDistiller().distill("hey how are you?", "I'm good thanks!", 4)
    assert len(ext) == 0


def test_source_turn_recorded():
    ext = RuleDistiller().distill("Remember that the API key rotates monthly.", "", 9)
    assert ext.facts[0].source_turn == 9


def test_dedup_within_extraction():
    ext = RuleDistiller().distill(
        "Remember that the office closes at 6. Also remember that the office closes at 6.",
        "",
        5,
    )
    assert len(ext.facts) == 1
