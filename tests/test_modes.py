"""Tests for presentation modes (minimal/task) and learning modes (usage-weighted)."""

import json
import tempfile

from agent_memory.context import SYSTEM_PROMPT
from agent_memory.core import MemoryEngine
from agent_memory.storage import MemoryEntry, MemoryStore


def _engine(**kw):
    defaults = dict(state_dir=tempfile.mkdtemp(), memory_cap_tokens=150)
    defaults.update(kw)
    return MemoryEngine(**defaults)


# --- usage tracking ----------------------------------------------------------

def test_usage_tracked_from_assistant_reply():
    eng = _engine()
    eng.process_turn("Remember that the deploy runs every Friday at 9am.", "Sure, noted.")
    entry = eng.store.all("fact")[0]
    assert entry.uses == 0  # reply did not reference it

    # Answer that references the fact marks it used.
    eng.process_turn(
        "When is the deploy?",
        "The deploy runs every Friday at 9am.",
    )
    entry = eng.store.get(entry.id)
    assert entry.uses == 1
    assert entry.last_used_at != ""


def test_usage_report_orders_by_reference_count():
    eng = _engine()
    eng.process_turn("Remember that the API key rotates monthly.", "ok")
    eng.process_turn("What security thing matters?", "The API key rotates monthly, so plan for it.")
    api_entry = eng.store.all("fact")[0]
    eng.process_turn("Remember the office closes at 6pm.", "ok")

    report = eng.usage_report()
    assert report[0]["text"] == api_entry.text
    assert report[0]["uses"] >= 1
    assert any(r["uses"] == 0 for r in report)


def test_usage_weighted_eviction_protects_used_entries():
    eng = _engine(memory_cap_tokens=120)
    # Fill memory with many facts; mark only one as "used".
    for i in range(20):
        eng.process_turn(f"Remember that the project codename is alpha-{i} it is.", "ok")
    used = eng.store.all("fact")[2]
    used.mark_used()
    eng.store.save()
    eng.archive_if_needed(policy="usage")

    survivors = eng.store.all("fact")
    assert eng.memory_tokens() <= 120
    assert used.id in {e.id for e in survivors}  # referenced entry survived
    # Everything else evicted before the used one.
    assert all(e.uses <= used.uses for e in survivors)


def test_oldest_policy_still_available():
    eng = _engine(memory_cap_tokens=120)
    first_id = None
    for i in range(20):
        eng.process_turn(f"Remember that the project codename is alpha-{i} it is.", "ok")
        if i == 0:
            first_id = eng.store.all("fact")[0].id
    eng.store.all("fact")[5].mark_used()  # used, but 'oldest' ignores usage
    eng.archive_if_needed(policy="oldest")

    survivors = eng.store.all("fact")
    assert first_id not in {e.id for e in survivors}  # oldest evicted despite usage


# --- presentation modes ------------------------------------------------------

def test_minimal_mode_excludes_assistant_output():
    eng = _engine()
    eng.process_turn("Remember I prefer dark mode.", "Sure, dark mode.")
    ctx = eng.build_context("What do I prefer?", mode="minimal")
    assert "MEMORY" in ctx.user_prompt
    assert "Sure, dark mode" not in ctx.user_prompt
    assert "WORKING MEMORY" not in ctx.user_prompt


def test_task_mode_includes_bounded_working_memory():
    eng = _engine()
    for i in range(6):
        eng.process_turn(f"question {i}", f"answer {i}")
    ctx = eng.build_context("next", mode="task", task_window=3)
    prompt = ctx.user_prompt
    assert "WORKING MEMORY" in prompt
    # Last 3 turns present verbatim (user AND assistant)...
    assert "question 5" in prompt and "answer 5" in prompt
    assert "question 3" in prompt and "answer 3" in prompt
    # ...earlier turns NOT in the bounded window.
    assert "question 0" not in prompt and "answer 0" not in prompt
    assert "MEMORY" in prompt


def test_task_mode_default_window_and_system_unchanged():
    eng = _engine()
    eng.process_turn("Remember X is true.", "ok")
    ctx = eng.build_context("next", mode="task")
    assert ctx.system == SYSTEM_PROMPT
    assert ctx.prompt_tokens > 0


def test_invalid_mode_rejected():
    eng = _engine()
    try:
        eng.build_context("x", mode="deep")
        assert False, "should have raised"
    except ValueError:
        pass


# --- persistence / backward compat ------------------------------------------

def test_v1_state_without_usage_fields_loads():
    store = MemoryStore(tempfile.mkdtemp())
    store.add(MemoryEntry(text="old fact", kind="fact"))
    store.save()
    # Simulate a v1 file: strip the new usage fields.
    data = json.loads(store.state_path.read_text())
    data["version"] = 1
    for e in data["entries"]:
        e.pop("uses", None)
        e.pop("last_used_at", None)
    store.state_path.write_text(json.dumps(data))

    store2 = MemoryStore(store.root)
    assert len(store2.all()) == 1
    entry = store2.all()[0]
    assert entry.uses == 0
    assert entry.last_used_at == ""
    assert entry.text == "old fact"