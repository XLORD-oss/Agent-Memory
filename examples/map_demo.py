"""Memory-as-map demo: profile, principles, arguments, perspectives, backlinks.

Run:  python examples/map_demo.py
"""

import tempfile

from agent_memory import MemoryEngine


def section(title: str, body: str = "") -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)
    if body:
        print(body)


def main() -> None:
    eng = MemoryEngine(state_dir=tempfile.mkdtemp(prefix="mapdemo-"))

    # ---- 1. the user profile ---------------------------------------------
    eng.add_profile_entry("identity", "researcher: nonlinear dynamics applied to weather and climate.")
    eng.add_profile_entry("domain", "chaos theory, predictability, three-body problem.")
    eng.add_profile_entry("style", "compact quantitative answers; phase portraits over prose.")
    eng.add_profile_entry("constraint", "no meetings before 10am; results as tables.")
    eng.add_profile_entry("goal", "build a compact-memory framework and validate it on real research sessions.")

    # ---- 2. first principles (axioms, never evicted) ---------------------
    eng.add_principle("Forecast skill is bounded by initial-condition uncertainty.", domain="chaos")
    eng.add_principle("In a chaotic system, close trajectories diverge exponentially.", domain="chaos")
    eng.add_principle("The model is not the bottleneck; the state you hand it is.", domain="framework")

    # ---- 3. facts (the map) ----------------------------------------------
    lorenz = eng.add_entry("Lorenz-63 used as the toy model for monsoon predictability.", kind="fact", domain="chaos")
    benettin = eng.add_entry("Lyapunov spectrum via Benettin with QR reorthonormalization every 10 steps.", kind="fact", domain="numerics")
    embed = eng.add_entry("Embedding: FNN saturates at m=7, Theiler window 40, tau from autocorrelation.", kind="fact", domain="numerics")

    # ---- 4. arguments (claims derived from linked premises) ---------------
    arg1 = eng.add_argument(
        "Ensemble mean beats the control because it filters the unstable directions.",
        premises=[lorenz.id, benettin.id],
        domain="chaos",
    )
    arg2 = eng.add_argument(
        "The delay embedding is faithful only if the window is long enough.",
        premises=[embed.id],
        domain="chaos",
    )

    # ---- 5. perspectives (conflicting viewpoints on an open question) -----
    eng.add_perspective("Use Postgres for ensemble storage?", "for",
                        "handles the reanalysis joins cleanly at scale.",
                        links=[lorenz.id])
    eng.add_perspective("Use Postgres for ensemble storage?", "against",
                        "ops burden for a single-user research setup.",
                        links=[arg1.id])
    eng.add_perspective("Use Postgres for ensemble storage?", "open",
                        "decide after the benchmark.",
                        links=[arg2.id])

    section("Rendered memory files")
    print(eng.memory_md)
    print(eng.principles_md)
    print(eng.perspectives_md)
    print(eng.profile_md)

    section("The assembled prompt (fresh chat)")
    ctx = eng.build_context("What database should I use, and why?")
    print(ctx.user_prompt)
    print(f"\n[prompt tokens: {ctx.prompt_tokens} · memory tokens: {eng.memory_tokens()}]")

    section("Navigating the map (traversal)")
    arg = eng.store.all("argument")[0]
    print(f"argument: {arg.text}")
    for rel in eng.related(arg.id):
        print(f"  → related entry: {rel.text}  [{rel.kind}]")


if __name__ == "__main__":
    main()