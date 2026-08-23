"""Storage tests: persistence round-trip, markdown rendering, archiving."""

import tempfile
from pathlib import Path

from agent_memory.storage import MemoryEntry, MemoryStore


def _tmp_store():
    return MemoryStore(tempfile.mkdtemp())


def test_round_trip():
    store = _tmp_store()
    store.add(MemoryEntry(text="User works at NASA", kind="fact", source_turn=1))
    store.add(MemoryEntry(text="We decided to use Python", kind="conclusion", source_turn=2))
    store.save()

    store2 = MemoryStore(store.root)
    assert len(store2.all()) == 2
    assert store2.all("fact")[0].text == "User works at NASA"
    assert store2.all("conclusion")[0].kind == "conclusion"


def test_markdown_rendering_and_files():
    store = _tmp_store()
    store.add(MemoryEntry(text="Lives in Berlin", kind="fact", source_turn=1))
    store.add(MemoryEntry(text="Prefers concise answers", kind="preference", source_turn=2))
    store.add(MemoryEntry(text="Concluded X", kind="conclusion", source_turn=3))

    md = store.render_markdown(kinds=["fact"])
    assert "Lives in Berlin" in md
    assert "Concluded X" not in md

    memory_path = store.write_memory_md()
    perspectives_path = store.write_perspectives_md()
    assert Path(memory_path).exists()
    assert Path(perspectives_path).exists()
    assert "Concluded X" in perspectives_path.read_text()
    assert "Prefers concise answers" in perspectives_path.read_text()


def test_archive_moves_entries_out():
    store = _tmp_store()
    e1 = store.add(MemoryEntry(text="fact one", kind="fact", source_turn=1))
    e2 = store.add(MemoryEntry(text="conclusion one", kind="conclusion", source_turn=2))
    store.save()

    moved = store.archive_entries([e1.id], era="2026-08-23")
    assert moved == 1
    assert store.get(e1.id) is None
    assert store.get(e2.id) is not None
    archive_files = list(store.archive_dir.glob("*.md"))
    assert len(archive_files) == 1
    assert "fact one" in archive_files[0].read_text()


def test_token_overlap_similarity():
    a = MemoryEntry(text="User prefers dark mode", kind="preference")
    b = MemoryEntry(text="User prefers dark mode and blue accents", kind="preference")
    c = MemoryEntry(text="The weather is sunny today", kind="fact")
    assert a.token_overlap(b) > 0.5
    assert a.token_overlap(c) < 0.2


def test_invalid_kind_rejected():
    try:
        MemoryEntry(text="x", kind="nonsense")
        assert False, "should have raised"
    except ValueError:
        pass
