"""Unified-policy demo: one scoring function, tuned by task.

Shows the SAME engine producing either a pure fresh-chat context or a
coding working-memory context, purely by swapping the policy vector —
no special modes, just weights.

Run:  python examples/policy_demo.py
"""

import tempfile

from agent_memory import CODING, GENERAL, MemoryPolicy, MemoryEngine


def show(title: str, ctx) -> None:
    print(f"\n{'=' * 60}\n{title}  [{ctx.mode} profile, {ctx.budget_tokens} token budget, "
          f"{ctx.used_tokens} used]\n{'=' * 60}")
    print(ctx.user_prompt)


def main() -> None:
    eng = MemoryEngine(state_dir=tempfile.mkdtemp(prefix="policydemo-"))

    # Some durable facts + a recent coding exchange.
    eng.process_turn("Remember that my stack is Python + FastAPI and it deploys to Fly.io.", None)
    eng.process_turn("Remember that I hate meetings before 10am.", None)
    eng.process_turn("We decided to use Postgres for the new service.", None)
    eng.process_turn("fix the import error in main.py", "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```")
    eng.process_turn("the route returns 500, check the query", "def get_user(id):\n    user = db.query(id)\n    return user")

    show("GENERAL (fresh chat) — assistant replies never replayed", eng.build_context("what should I fix next?"))
    show("CODING (working memory) — recent code turns are task context", eng.build_context("what should I fix next?", profile="coding"))
    show("CODING + explicit priority — facts you care about always win",
         eng.build_context("what should I fix next?", profile="coding", budget=180))

    print("\n-- eviction under pressure: coding profile keeps code-tagged facts --")
    eng2 = MemoryEngine(state_dir=tempfile.mkdtemp(prefix="policydemo2-"), policy=CODING, memory_cap_tokens=100)
    for i in range(30):
        eng2.process_turn(f"Remember that the project codename is alpha-{i} and it matters for the API design.", None)
    survivors = [e.text for e in eng2.store.all()]
    print(f"survived {len(survivors)} entries within cap {eng2.memory_cap_tokens} tokens:")
    for s in survivors:
        print("  -", s[:70])

    print("\n-- full custom policy (first principles) --")
    custom = MemoryPolicy(
        kind_weights={"fact": 2.0, "conclusion": 1.5, "preference": 1.0,
                      "user_turn": 0.3, "assistant_turn": 0.4},  # assistant FORGIVEN at 0.4
        priority_weight=2.0,          # user priorities dominate
        usage_weight=0.1,             # usage barely matters
        recency_weight=0.05,          # age barely matters
        affinity_weight=0.5,          # task affinity matters
        half_life_turns=100,
        budget_tokens=800,
        task_window_turns=5,
    )
    eng.set_policy(custom)
    ctx = eng.build_context("what should I fix next?")
    print("profile recorded as:", ctx.mode, "| used tokens:", ctx.used_tokens, "of", ctx.budget_tokens)
    print(ctx.user_prompt[:400], "...")


if __name__ == "__main__":
    main()