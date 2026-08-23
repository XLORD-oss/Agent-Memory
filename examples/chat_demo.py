"""A minimal chat loop showing memory grow slowly while the transcript grows fast.

Default runs offline with the mock model; pass --model for a real OpenAI-compatible
model. Watch: memory stays small and the assistant's own past answers never appear
in the context it sees.

    python examples/chat_demo.py
    python examples/chat_demo.py --model gpt-4o-mini
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_memory import MemoryEngine, MockModel, OpenAICompatClient
from agent_memory.distiller import RuleDistiller
from agent_memory.tokens import estimate_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal chat demo with compact memory")
    parser.add_argument("--model", default=None, help="real model name (offline mock otherwise)")
    parser.add_argument("--state-dir", default=None, help="where to persist memory")
    args = parser.parse_args()

    client = (
        OpenAICompatClient(model=args.model)
        if args.model
        else MockModel(mode="sycophancy", seed=1)
    )
    engine = MemoryEngine(
        state_dir=args.state_dir or tempfile.mkdtemp(prefix="chatdemo-"),
        distiller=RuleDistiller(),
    )

    print("Agent-Memory chat demo. Type 'quit' to exit.")
    print("Tips: say 'remember that ...', 'I prefer ...', 'we decided ...'")
    while True:
        try:
            user = input("\nYou: ").strip()
        except EOFError:
            break
        if not user or user.lower() in ("quit", "exit"):
            break

        ctx = engine.build_context(user)
        reply = client.complete(ctx.to_messages())
        engine.process_turn(user, assistant_text=reply)

        print(f"Agent: {reply}")
        print(
            f"  [memory: {engine.entry_count} entries / {engine.memory_tokens()} tokens; "
            f"transcript so far: {estimate_tokens(' '.join(t['user'] for t in engine._raw_turns))} tokens]"
        )

    print("\nFinal memory:")
    print(engine.memory_md)


if __name__ == "__main__":
    main()
