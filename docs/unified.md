# The Unified Memory Policy — first principles

The three "modes" (minimal / task / segmented) were a leaky abstraction. They
are not modes; they are **points on one continuous surface**. This document is
the first-principles derivation of the surface, and the recipe for customizing
it until it matches *your* utilization.

## The first principle: context is a token budget

Every turn, the model gets a fixed budget of context tokens. Inside that budget
we must place as much *value* as possible. Value is not binary ("keep this,
drop that") — it is a score, and everything competes.

## The second principle: every unit of memory is a candidate

There is only one kind of thing in the system: **units** — a distilled fact, a
conclusion, a preference, a raw user turn, a raw assistant turn. Each unit has
the same five signals:

| Signal | Meaning | Where it comes from |
|---|---|---|
| `priority` | explicit user/system importance | `set_priority`, `prioritize()` on the fact; 1.0 neutral, >1 boosted, pin = unbounded |
| `usage` | how often the model's answers actually referenced it | zero-API-cost token-overlap tracking (`_track_usage`) |
| `recency` | how fresh it is | age in turns, exponential decay with `half_life_turns` |
| `affinity` | how on-task it is right now | `policy.affinity()` — tags ⊕ `boost_tags`, text ⊕ `boost_tokens` |
| `kind` | what it is | `fact` / `conclusion` / `preference` / `user_turn` / `assistant_turn` |

## The third principle: one scoring function

```python
value = kind_weight[kind] * (
      priority_weight * priority
    + usage_weight    * log(1 + uses)
    + recency_weight  * exp(-age / half_life_turns)
    + affinity_weight * affinity
)
```

Assembly = **greedily fill the budget with the highest-value units**.
Retention = **evict the lowest-value units when over the cap** (pinned and
decisions protected). The same function runs both directions — what the model
sees and what the memory keeps are the same optimization.

## What falls out for free

* **Fresh-chat is a point, not a mode.** Set
  `kind_weights["assistant_turn"] = 0.0` and raw assistant outputs score zero →
  they are *never* selected, even when the budget has room
  (`select_by_score` hard-skip on `score <= 0`). The model cannot anchor on
  "I already said X" because its raw past words are not there.

* **"Remembered stronger for coding tasks" is a weight, not a hack.** Give the
  coding profile `assistant_turn ≈ 0.9` (recent code turns become working
  memory), `affinity_weight ≈ 0.4`, `boost_tags = {code, api, repo, ...}` —
  and code-tagged facts beat general facts for the budget and in eviction,
  automatically. Non-code facts get archived first *because they score lower*,
  not because of a hardcoded rule.

* **What the model actually uses is what survives.** Usage is part of the
  score, so an entry your sessions keep referencing climbs above dead weight of
  any age. Memory converges to your real footprint (`usage_report()` shows it).

## The customization surface

Everything is a number on the vector. There is no branching.

```python
MemoryPolicy(
    kind_weights={...},      # what kinds earn tokens (set assistant_turn 0..1)
    priority_weight=...,     # how much explicit priority matters
    usage_weight=...,        # how much behavioral usage matters
    recency_weight=...,      # how much freshness matters
    affinity_weight=...,     # how much on-task-ness matters
    half_life_turns=...,     # recency decay speed
    budget_tokens=...,       # context budget for memory+perspectives+turns
    task_window_turns=...,   # how many recent raw turns are eligible
    protect_conclusions=..., # never evict decisions
    boost_tags=...,          # tag ⇒ affinity 1.0
    boost_tokens=...,        # substring ⇒ affinity 1.0
)
```

Presets that ship (see `policy.py`):

| Profile | assistant_turn | task window | affinity | Built for |
|---|---|---|---|---|
| `general` | 0.0 | 0 | 0.0 | fresh chat, advice, synthesis |
| `coding` | 0.9 | 12 | 0.4 | code, debugging, architecture |
| `research` | 0.15 | 6 | 0.3 | analysis, derivations, literature |
| `writing` | 0.25 | 6 | 0.2 | tone, structure, editing |

Use `profile="auto"` to let `detect_profile()` pick from the current message;
or skip profiles entirely and pass a fully custom `policy=`.

## Recipies for real customization

```python
# I do research; facts matter, assistant reasoning lightly
eng = MemoryEngine(policy=RESEARCH, memory_cap_tokens=4000)

# Everything about my stack is sacred
eng.prioritize("fastapi", 10.0)
eng.pin("the-entry-id-...")          # never evicted, always in context

# A coding session: recent code turns stay visible, code facts win eviction
ctx = eng.build_context("debug the 500", profile="coding", budget=2400)

# Mid-conversation task switch is just a vector swap
eng.set_policy_profile("coding")

# Zero-assistant raw, EVER, even with a window: raise and override
strict = CODING  # start from coding
strict.kind_weights["assistant_turn"] = 0.0
eng.set_policy(strict)
```

## The knobs, stated honestly

* **Budget too small** → working memory starves; long derivations lose context.
* **assistant_turn too high** → anchoring pressure returns (the sycophancy
  benchmark is waiting to measure it).
* **priority_weight ≫ usage_weight** → memory reflects what you *said* matters,
  not what you *use*. Usually you want both.
* **recency_weight too high** → memory churns: old-but-valuable facts fade.
* **affinity_weight too high** → context becomes task-narrow; cross-domain facts
  get squeezed out.

There is no free lunch — but now the lunch is *your recipe*. `analyze_trace.py`
(next on the roadmap) replays your real logs through any policy and shows the
token/value curve, so the knobs get tuned on evidence, not vibes.