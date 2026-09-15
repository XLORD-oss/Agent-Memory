<div align="center">

# Agent-Memory

**Compact, self-maintaining memory for LLM agents.**
Near-linear token cost · no context rot · less self-anchoring.

*Not a memory library that just stores things — a memory layer that changes what
the model is asked to work with, and ships the benchmarks to prove the gap.*

</div>

---

## The problem: full-transcript replay is the default, and it's broken

Every long-running agent defaults to replaying the whole raw conversation into
context each turn. That one decision causes three independent, compounding
failure modes:

| # | Failure mode | The evidence |
|---|---|---|
| 1 | **Context rot** — accuracy falls as context grows, well before the window is full | Chroma tested 18 frontier models (GPT-4.1, Claude 4, Gemini 2.5, Qwen3…): *every* model degraded with input length — 30–50% accuracy drops from clutter on tasks the models solve trivially when context is clean ([research.trychroma.com/context-rot](https://research.trychroma.com/context-rot)) |
| 2 | **Quadratic token cost** — turn N carries ~N turns of tokens | Arithmetic: total tokens processed over T turns = `s·T·(T+1)/2` → **O(T²)**. A 1,000-turn chat at ~500 tokens/turn replays ~250M tokens |
| 3 | **Self-anchoring & sycophancy** — the model re-reads its own past answers and treats "I already said X" as evidence | Self-Anchoring Calibration Drift ([arXiv:2603.01239](https://arxiv.org/abs/2603.01239)); RLHF amplifies sycophancy and bigger models show *more* of it ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)) |

The sharpened framing: **the models are fully capable — the failure is that the
working context doesn't stay clean.** Capability doesn't move; *expression* does.
This framework removes the failure mode, it doesn't upgrade the weights.

## The idea: a compact memory that replaces replay

Instead of the raw transcript, the agent keeps a small set of curated files that
are the **only** memory the model ever sees:

```
memory.md          # distilled facts — grows ONLY when a genuinely new fact appears
perspectives.md    # conclusions & preferences — "concluded X", without the drama that built up to it
archive/           # older entries split off by era — out of the active context
```

```
raw turn ─> INGEST ─> DISTILL ─> MERGE ─> ARCHIVE ─> BUILD CONTEXT ─> model
                │     (facts /        (dedupe,   (keep active   (memory + perspectives +
                │      conclusions /    commit)    memory under   recent USER turns ONLY)
                │      preferences)                the cap)
                └─ raw log kept for audit, NEVER replayed
```

Two properties fall out of the design:

* **Each turn is a fresh chat.** Prior *assistant* outputs are never placed back
  in context — the model can't anchor on its own past words. It sees the distilled
  "concluded X", not the social momentum that surrounded it.
* **Memory is human-readable.** It's Markdown, not a black box — read it, correct
  it, trust it.

**Memory as a map** ([docs/map.md](docs/map.md)) — not a flat list, a
navigable graph:

| Kind | Sym | Meaning |
|---|---|---|
| profile | ▣ | user profile — identity / domain / style / constraints (always in context) |
| principle | ▲ | first principles / axioms — **never evicted** |
| fact | • | durable knowledge |
| conclusion | ◆ | settled decisions — **never evicted** |
| argument | ⇒ | claims derived from linked premises |
| perspective | ◉ | for/against/open viewpoints on open questions, with links |

```python
eng.add_profile_entry("domain", "chaos theory, predictability, three-body problem")
eng.add_principle("Forecast skill is bounded by initial-condition uncertainty.")
eng.add_argument("Ensemble mean beats control — it filters unstable directions.",
                 premises=[lorenz.id, benettin.id])
eng.add_perspective("Use Postgres?", "against", "ops burden for a single-user setup.")
eng.link(claim.id, premise.id)     # edges — the map is traversable
```

**Unified policy** ([docs/unified.md](docs/unified.md)) — one scoring function
drives both what enters context and what stays in memory:

```python
value = kind_weight · (priority_weight·priority
                       + usage_weight·log(1+uses)
                       + recency_weight·decay(age)
                       + affinity_weight·task_affinity)
```

Context is a budget, filled highest-value-first. Fresh-chat is just the policy
where `assistant_turn = 0` (raw replies never score); coding working-memory is
the policy where it's `0.9`. Everything is a weight, so it's fully tunable:

```python
eng.set_policy_profile("coding")      # presets: general | coding | research | writing
eng.build_context(msg, profile="auto")# or auto-detect from the message
eng.set_priority(entry_id, 10.0)      # explicit priority on a fact
eng.prioritize("fastapi", 10.0)       # remember a topic stronger
eng.pin(entry_id)                     # unbounded priority — never evicted
eng.set_policy(MemoryPolicy(...))     # full custom vector (first principles)
```

The old `minimal`/`task` modes remain as sugar for the presets. Every policy
keeps the anti-rot core: the full raw transcript is never replayed.

## What we claim (and don't)

| Claim | Status |
|---|---|
| Compact memory removes the measured 30–50% context-rot failure mode | **Evidence-backed** (Chroma, 18 models; Lost in the Middle, Liu et al.) |
| Token cost goes from O(T²) to ~O(T) | **Arithmetic** — see `benchmarks/token_cost/` |
| Keeping prior outputs out of context reduces self-anchoring/sycophantic flipping | **Hypothesis to prove** — the benchmark ships with the repo |

We do **not** claim this makes models smarter. It makes existing capability
*express* reliably. And "drastically reduces bias" is the claim you get to prove
with `benchmarks/sycophancy/`, not a finding you can cite yet.
See [docs/claim.md](docs/claim.md) for the precise statement and
[docs/evidence.md](docs/evidence.md) for the citation bank.

## Quickstart

```bash
pip install -e .            # from repo root (no hard deps; [llm] for model calls)
python examples/quickstart.py
```

```python
from agent_memory import MemoryEngine

engine = MemoryEngine(state_dir=".agent-memory")
engine.process_turn("Remember that I work at NASA and research chaos theory.")
engine.process_turn("I prefer concise answers.", "Got it!")

ctx = engine.build_context("What do you know about me?")
print(ctx.user_prompt)          # memory + perspectives + recent user turns only
print(ctx.prompt_tokens)        # small and bounded, no matter how long the chat gets
```

## Benchmarks — two falsifiable experiments + one arithmetic table

All three run **offline** with the mock model (CI-friendly) and against **real
models** with `--model` — same task, same model, two conditions: full raw
transcript vs. the compact memory file.

```bash
# 1. Context-rot: same task, two conditions, measure the accuracy gap
python -m benchmarks.context_rot.run --mock            # → raw 30% / memory 100% (mock)
python -m benchmarks.context_rot.run --model gpt-4o-mini --turns 200 --facts 10

# 2. Sycophancy FlipFlop: answer → push back → does it flip? (SYCON ToF/NoF)
python -m benchmarks.sycophancy.run_flipflop --mock    # → full 100% flip / memory 0% (mock)
python -m benchmarks.sycophancy.run_flipflop --model gpt-4o-mini --rounds 4

# 3. Token cost: the O(T²) vs O(T) table, no model needed
python -m benchmarks.token_cost.model

# 4. Analyze YOUR usage: replay a real chat log through any policy, offline
python -m agent_memory.analysis --trace logs/session.jsonl --compare
python -m agent_memory.analysis --trace demo --repeat 5 --plot cost.png
```

Every benchmark is built on a **published methodology** (Chroma's focused-vs-full
design; SYCON Bench's Turn-of-Flip / Number-of-Flip metrics), not a home-grown
eval — so a positive number is a result you can stand behind, not a pitch.
See [benchmarks/README.md](benchmarks/README.md).

## Repository layout

```
src/agent_memory/       # the library
  core.py               #   MemoryEngine: ingest → distill → merge → archive → context
  storage.py            #   file-backed memory.md / perspectives.md / archive/
  distiller.py          #   rules-based + LLM-based extraction
  tokens.py             #   token estimation + the cost model
  context.py            #   the fresh-chat prompt contract
  llm.py                #   OpenAI-compatible client + offline mock of failure modes
  export.py             #   paired SFT data: same targets, four context contracts (docs/training.md)
benchmarks/             # the three experiments (mock or real model)
docs/                   # claim.md · evidence.md · architecture.md · collaboration.md · roadmap.md
examples/               # quickstart.py · chat_demo.py
tests/                  # 30 tests, all offline
```

## Evidence in one line

> 18 frontier models degrade 30–50% on cluttered contexts; replay costs O(T²);
> models re-read their own answers and cave under social pressure. Agent-Memory
> replaces replay with a compact, curated store — and ships the benchmarks to
> measure how much that recovers.

More: [docs/evidence.md](docs/evidence.md) · [docs/architecture.md](docs/architecture.md) · [docs/collaboration.md](docs/collaboration.md) (budget, credits, co-authorship) · [docs/training.md](docs/training.md) (fine-tuning under the contract) · [docs/kaggle.md](docs/kaggle.md) (run the pilot on a free T4 ×2) · [docs/models.md](docs/models.md) (which models, which lane) · [docs/proposal.md](docs/proposal.md) (the university ask) · [docs/roadmap.md](docs/roadmap.md)

## License

MIT
