"""Tests for the unified memory policy: priorities, affinity, budgets, pinning.

The unified model: one scoring function (priority · usage · recency · affinity,
weighted per kind) drives BOTH context selection and retention. Fresh-chat is
the point where assistant_turn weight is 0; coding working-memory is the point
where it is high. These tests pin that down.
"""

import tempfile

from agent_memory import MemoryPolicy
from agent_memory.core import MemoryEngine
from agent_memory.policy import GENERAL, CODING, detect_profile, score_item, select_by_score
from agent_memory.tokens import estimate_tokens


def _engine(**kw):
    defaults = dict(state_dir=tempfile.mkdtemp())
    defaults.update(kw)
    return MemoryEngine(**defaults)


# --- the scoring function ----------------------------------------------------

def test_score_item_priority_dominates():
    base = score_item("x", "fact", priority=1.0, uses=0, age_turns=100, policy=GENERAL)
    boosted = score_item("x", "fact", priority=10.0, uses=0, age_turns=100, policy=GENERAL)
    assert boosted > base
    recent = score_item("x", "fact", priority=1.0, uses=0, age_turns=0, policy=GENERAL)
    assert recent > base  # recency decay means old items lose value


def test_score_item_usage_and_recency_terms():
    used = score_item("x", "fact", 1.0, uses=50, age_turns=0, policy=GENERAL)
    unused = score_item("x", "fact", 1.0, uses=0, age_turns=0, policy=GENERAL)
    assert used > unused


def test_assistant_turn_zero_in_general():
    assert GENERAL.kind_weights["assistant_turn"] == 0.0
    s = score_item("answer", "assistant_turn", 1.0, 0, 0, GENERAL)
    assert s == 0.0  # never selected in fresh-chat


def test_coding_profile_boosts_assistant_and_affinity():
    assert CODING.kind_weights["assistant_turn"] > 0
    code_fact = score_item("stack: python + fastapi", "fact", 1.0, 0, 100,
                           CODING, affinity=1.0)
    plain_fact = score_item("stack: python + fastapi", "fact", 1.0, 0, 100,
                            CODING, affinity=0.0)
    assert code_fact > plain_fact


def test_select_by_score_greedy_budget():
    cands = [
        {"score": 1.0, "tokens": 100, "id": "a"},
        {"score": 5.0, "tokens": 100, "id": "b"},
        {"score": 3.0, "tokens": 100, "id": "c"},
    ]
    picked = select_by_score(cands, budget_tokens=200)
    assert [c["id"] for c in picked] == ["b", "c"]  # greedy: b + c fit the budget


def test_select_by_score_never_selects_zero_weight():
    cands = [
        {"score": 1.0, "tokens": 100, "id": "fact"},
        {"score": 0.0, "tokens": 10, "id": "assistant"},  # fresh-chat: weight 0
    ]
    picked = select_by_score(cands, budget_tokens=500)
    assert [c["id"] for c in picked] == ["fact"]


# --- context building under profiles ------------------------------------------

def test_general_profile_still_fresh_chat_keeps_flow():
    eng = _engine(recent_window=2)
    eng.process_turn("Remember I like dark mode.", "ok")
    eng.process_turn("What do I prefer?", "dark mode")
    ctx = eng.build_context("next", profile="general")
    prompt = ctx.user_prompt
    assert "dark mode" in prompt.lower()       # memory + flow present
    assert "WORKING MEMORY" not in prompt      # no assistant raw output
    assert "ok" not in prompt


def test_coding_profile_includes_working_memory():
    eng = _engine()
    for i in range(6):
        eng.process_turn(f"fix bug number {i}", f"def fix_{i}(): pass")
    ctx = eng.build_context("keep going", profile="coding", task_window=3)
    prompt = ctx.user_prompt
    assert "WORKING MEMORY" in prompt
    assert "def fix_5(): pass" in prompt       # recent assistant output visible
    assert "def fix_0(): pass" not in prompt   # outside the bounded window


def test_auto_profile_detection():
    assert detect_profile("fix the import error in main.py") == "coding"
    assert detect_profile("derive the chaos model equation") == "research"
    assert detect_profile("edit the tone of this essay") == "writing"
    assert detect_profile("hello, how are you?") == "general"


def test_auto_profile_in_build_context():
    eng = _engine()
    eng.process_turn("help me debug", "ok")
    # Auto-detection switches the profile to coding mid-conversation, which
    # makes recent assistant turns eligible as working memory.
    ctx = eng.build_context("can you fix the TypeError in the repo?", profile="auto")
    assert ctx.mode == "coding"
    assert "WORKING MEMORY" in ctx.user_prompt
    assert "ok" in ctx.user_prompt  # the assistant reply is now task context


def test_custom_policy_fully_customizable():
    # A policy that allows a big assistant window but zero assistant weight:
    # user turns flow, assistant outputs still never replayed.
    custom = MemoryPolicy(kind_weights={
        "fact": 1.0, "conclusion": 1.0, "preference": 1.0,
        "user_turn": 0.6, "assistant_turn": 0.0,
    }, task_window_turns=10)
    eng = _engine()
    eng.process_turn("question a", "assistant a")
    ctx = eng.build_context("next", policy=custom)
    assert "assistant a" not in ctx.user_prompt


# --- priorities and pinning ---------------------------------------------------

def test_priority_wins_context_budget():
    eng = _engine(memory_cap_tokens=200)
    eng.process_turn("Remember that the API key is rotated on the 1st of each month.", "ok")
    eng.process_turn("Remember that the office has a small green couch by the window.", "ok")
    eng.process_turn("Remember that the team uses a blue sticky note system for standups.", "ok")

    api = [e for e in eng.store.all() if "API key" in e.text][0]
    eng.set_priority(api.id, 50.0)  # strongly remembered

    sorted_by_score = sorted(
        eng.store.all(),
        key=lambda e: score_item(e.text, e.kind, e.effective_priority(), e.uses, 0,
                                 eng.policy, affinity=0.0),
        reverse=True,
    )
    assert sorted_by_score[0].id == api.id


def test_pin_protects_from_eviction():
    eng = _engine(memory_cap_tokens=60)
    # Pin the first entry immediately (before archive pressure builds).
    eng.process_turn("Remember that the project tracer value is gamma-0.", "ok")
    victim_id = eng.store.all("fact")[-1].id
    eng.pin(victim_id)
    # Then flood memory well past the cap.
    for i in range(1, 25):
        eng.process_turn(f"Remember that the project tracer value is gamma-{i}.", "ok")
    eng.archive_if_needed(policy="usage")

    assert eng.memory_tokens() <= 60
    assert eng.store.get(victim_id) is not None  # pinned survived


def test_pin_priority_effective():
    eng = _engine()
    eng.process_turn("Remember X is yellow.", "ok")
    e = eng.store.all("fact")[0]
    assert e.effective_priority() == 1.0
    eng.pin(e.id)
    assert eng.store.all("fact")[0].effective_priority() > 1e9
    eng.unpin(e.id)
    assert eng.store.all("fact")[0].effective_priority() == 1.0


def test_prioritize_by_text():
    eng = _engine()
    eng.process_turn("Remember that the deploy target is staging for now.", "ok")
    eng.process_turn("Remember that lunch is at 1pm.", "ok")
    n = eng.prioritize("deploy", 8.0)
    assert n == 1
    deploy = [e for e in eng.store.all() if "deploy" in e.text][0]
    assert deploy.priority == 8.0


def test_priority_persists_on_reload():
    eng = _engine()
    eng.process_turn("Remember that the demo date is the last Friday.", "ok")
    e = eng.store.all("fact")[0]
    eng.set_priority(e.id, 7.5)
    eng2 = MemoryEngine(state_dir=eng.store.root)
    reloaded = eng2.store.get(e.id)
    assert reloaded.priority == 7.5


def test_budget_respected_during_context_build():
    eng = _engine(memory_cap_tokens=200)
    for i in range(10):
        eng.process_turn(f"Remember that the long-standing project detail is milestone number {i} for the quarter.", "ok")
    ctx = eng.build_context("what milestone matters?", budget=120)
    assert ctx.used_tokens <= 120
    assert ctx.scored_candidates >= 10