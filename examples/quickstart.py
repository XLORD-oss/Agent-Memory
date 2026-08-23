"""Quickstart: see the compact memory engine in action in under a minute.

Run:  python examples/quickstart.py
"""

import tempfile

from agent_memory import (
    MemoryEngine,
    cost_table,
    estimate_tokens,
)
from agent_memory.context import Context


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    # 1. The token-cost claim, in one table.
    section("1. Token cost: full-history replay vs compact memory")
    print(cost_table())

    # 2. A long conversation, distilled into a small memory.
    section("2. Distilling a long conversation into compact memory")
    engine = MemoryEngine(state_dir=tempfile.mkdtemp(prefix="quickstart-"))

    script = [
        "Hi! Remember that I work at NASA and I'm researching chaos theory.",
        "My name is Ada. I prefer concise answers with bullet points.",
        "Also, please never schedule meetings before 10am.",
        "We decided to build the agent prototype in Python using an LLM.",
        "How's the weather today?",
        "Remember that the demo for the funder is on the last Friday of the month.",
        "We confirmed that the context-rot benchmark runs with the mock model.",
        "What color should the logo be?",
        "I prefer a dark theme for all our internal tools.",
        "We concluded that archiving old facts keeps the active memory small.",
    ]
    raw_tokens = 0
    for user in script:
        # The engine distills each turn; the "assistant" text is optional.
        engine.process_turn(user, assistant_text="(placeholder reply)")
        raw_tokens += estimate_tokens(user)

    print(f"raw conversation : {len(script):>3} turns, {raw_tokens:>5} tokens")
    print(f"compact memory   : {engine.entry_count:>3} entries, {engine.memory_tokens():>5} tokens "
          f"({raw_tokens / max(1, engine.memory_tokens()):.1f}x smaller)")
    print("\n--- memory.md ---")
    print(engine.memory_md)
    print("--- perspectives.md ---")
    print(engine.perspectives_md)

    # 3. The fresh-chat contract: prior assistant outputs are never replayed.
    section("3. Fresh-chat contract (the anti-self-anchoring mechanism)")
    ctx: Context = engine.build_context("What do you know about me?")
    user_prompt = ctx.user_prompt
    assert "(placeholder reply)" not in user_prompt  # assistant raw output absent
    print("Context sent to the model contains: MEMORY, PERSPECTIVES, recent USER turns,")
    print("and the current message. It does NOT contain any prior assistant reply.")
    print(f"\nPrompt size: {ctx.prompt_tokens} tokens "
          f"(vs ~{raw_tokens} raw tokens in the transcript)")

    print("\nDone. The memory is human-readable and editable in the state dir.")


if __name__ == "__main__":
    main()
