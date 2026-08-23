"""End-to-end MemoryEngine tests: the fresh-chat contract, dedupe, archive, cost."""

import tempfile

from agent_memory.core import MemoryEngine
from agent_memory.storage import MemoryEntry
from agent_memory.tokens import estimate_tokens


def _engine(**kw):
    return MemoryEngine(state_dir=tempfile.mkdtemp(), **kw)


def test_ingest_distill_merge_flow():
    eng = _engine()
    n = eng.process_turn("Hi! Remember that I work at NASA.", "Welcome!")
    assert n >= 1
    assert eng.entry_count >= 1
    assert "NASA" in eng.memory_md
    assert eng.raw_turn_count == 1


def test_duplicates_do_not_grow_memory():
    eng = _engine()
    eng.process_turn("Remember that the deploy is every Friday at 9am.", "ok")
    count_after_first = eng.entry_count
    eng.process_turn("Also, remember that the deploy is every Friday at 9am.", "ok")
    assert eng.entry_count == count_after_first  # deduplicated


def test_fresh_chat_contract_no_prior_assistant_output():
    eng = _engine()
    eng.process_turn("Remember that I prefer dark mode.", "Sure, dark mode it is.")
    eng.process_turn("What do I prefer?", "You prefer dark mode.")
    ctx = eng.build_context("What do I prefer?")
    user_prompt = ctx.user_prompt
    # Memory entry present...
    assert "dark mode" in user_prompt.lower()
    # ...but the assistant's raw replies are NOT replayed.
    assert "Sure, dark mode it is" not in user_prompt
    assert "You prefer dark mode" not in user_prompt


def test_perspectives_are_included():
    eng = _engine()
    eng.process_turn("We decided to migrate to Postgres.", "Noted.")
    ctx = eng.build_context("")
    assert "Postgres" in ctx.user_prompt


def test_archive_keeps_memory_bounded():
    eng = _engine(memory_cap_tokens=60)
    for i in range(30):
        eng.process_turn(f"Remember that the project codename is alpha-{i} and it matters.")
    assert eng.memory_tokens() <= 60
    assert len(list(eng.store.archive_dir.glob("*.md"))) >= 1


def test_compact_context_stays_small_while_transcript_grows():
    eng = _engine(memory_cap_tokens=400, recent_window=4)
    for i in range(40):
        eng.process_turn(f"Remember that fact number {i} is worth remembering for later.")
    last = eng.build_context("Question 39")
    compact_content = estimate_tokens(last.user_prompt)
    # A naive last turn would replay ~40 raw turns of ~40 tokens each.
    naive_last_turn = 40 * 40
    assert compact_content < naive_last_turn
    assert eng.memory_tokens() <= 400  # memory is bounded by the cap


def test_total_tokens_processed_includes_system_overhead_fairly():
    from agent_memory.context import SYSTEM_PROMPT
    from agent_memory.tokens import estimate_tokens, full_history_total_tokens

    turns = 40
    eng = _engine(memory_cap_tokens=400, recent_window=4)
    for i in range(turns):
        eng.process_turn(f"Remember that fact number {i} is worth remembering for later.")
    for i in range(turns):
        eng.build_context(f"Question {i}")
    total = eng.total_tokens_processed()
    assert total > 0
    # Full-history replay also pays the system prompt every turn, so include it.
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    naive = full_history_total_tokens(turns, per_turn=40) + turns * sys_tokens
    assert total < naive


def test_build_context_returns_messages():
    eng = _engine()
    eng.process_turn("Remember that my name is Ada.", "Nice to meet you, Ada!")
    ctx = eng.build_context("What is my name?")
    messages = ctx.to_messages()
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Ada" in messages[1]["content"]
    assert ctx.prompt_tokens > 0


def test_recent_window_keeps_flow():
    eng = _engine(recent_window=2)
    eng.process_turn("First.")
    eng.process_turn("Second.")
    eng.process_turn("Third.")
    ctx = eng.build_context("Fourth.")
    recent = ctx.recent_user_turns
    assert recent == ["Second.", "Third."]
