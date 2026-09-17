# The memory as a map

The compact memory is not a list — it is a **navigable graph** of what the model
knows, arranged the way you'd arrange a working notebook:

```
                 ┌─────────────┐
                 │ USER PROFILE │  who is working here (identity, domain, style,
                 └──────┬──────┘  constraints, goals) — always in context
                        │
                 ┌──────▼──────┐
                 │  PRINCIPLES │  first principles / axioms — ▲ never evicted
                 └──────┬──────┘
        ┌───────────────┼────────────────┐
        │               │                │
   ┌────▼────┐    ┌─────▼─────┐    ┌─────▼──────┐
   │  FACTS   │◄──►│ ARGUMENTS │◄──►│ PERSPECTIVE│  for/against/open on open
   └─────────┘    └───────────┘    └────────────┘  questions, each with links
```

## The seven kinds

| Kind | Sym | Meaning | Eviction |
|---|---|---|---|
| `fact` | • | durable knowledge (the map's nodes) | scored, evictable |
| `conclusion` | ◆ | a settled decision ("concluded X") | **never evicted** |
| `preference` | ★ | stable user value / style | evictable |
| `principle` | ▲ | a first principle / axiom | **never evicted** |
| `argument` | ⇒ | a claim that derives from linked premises | evictable |
| `perspective` | ◉ | a viewpoint (FOR/AGAINST/OPEN) on a question | evictable |
| `profile` | ▣ | user-profile field (identity/domain/style/constraint/goal) | high priority |

Protected kinds (`PROTECTED_KINDS = conclusion, principle`) are excluded from
eviction entirely — the axioms and the decisions survive any memory pressure.

## Edges (the map part)

Every entry can carry `links` — IDs of related entries. Rendering shows them as
backlinks:

```
- **FOR** ◉ handles the reanalysis joins cleanly at scale.  → «Lorenz-63 used as the toy …»
- ⇒ Ensemble mean beats the control because it filters the unstable directions.
    → «Lorenz-63 used as the toy …», «Lyapunov spectrum via Benettin …»
```

Navigable programmatically too: `engine.related(entry_id)` resolves an entry's
links into live entries; `engine.link(a.id, b.id)` draws an edge.

## How to build one (the engine API)

```python
# profile
eng.add_profile_entry("identity", "researcher: nonlinear dynamics applied to weather.")
eng.add_profile_entry("domain",  "chaos theory, predictability, three-body problem.")
eng.add_profile_entry("style",   "compact quantitative answers.")
eng.add_profile_entry("constraint", "no meetings before 10am.")

# first principles — baked in, never evicted
eng.add_principle("Forecast skill is bounded by initial-condition uncertainty.", domain="chaos")

# facts
lorenz = eng.add_entry("Lorenz-63 as the toy model for monsoon predictability.", kind="fact")

# logical arguments (claims with linked premises)
benettin = eng.add_entry("Lyapunov spectrum via Benettin with QR reorthonormalization.", kind="fact")
eng.add_argument(
    "Ensemble mean beats the control because it filters the unstable directions.",
    premises=[lorenz.id, benettin.id],
)

# perspectives — multiple coexisting viewpoints
eng.add_perspective("Use Postgres?", "for",     "handles reanalysis joins at scale.", links=[lorenz.id])
eng.add_perspective("Use Postgres?", "against", "ops burden for a single-user setup.")
eng.add_perspective("Use Postgres?", "open",    "decide after the benchmark.")
```

## What the model actually sees (assembled prompt)

```
SYSTEM
USER PROFILE        ← who you are, how you work (profile entries, field-labelled)
FIRST PRINCIPLES    ← your axioms (▲)
MEMORY              ← facts (•)
PERSPECTIVES        ← conclusions (◆), preferences (★), viewpoints (◉ FOR/AGAINST/OPEN)
ARGUMENTS           ← derived claims (⇒)
WORKING MEMORY      ← recent turns, only if the task policy allows
CURRENT USER MESSAGE
```

Everything still goes through the unified scoring policy and token budget, so
the map structure costs nothing until it earns its tokens. Principles and
profile have higher base weights (they're the foundation), which is how they
usually make the cut while trivia facts compete.

## Why this matters for research work

* **Multiple perspectives, held simultaneously** — the memory stops force-flattening
  an open question into a single answer. "Use Postgres?" can be FOR *and* AGAINST
  until you've decided. This is the difference between a memory that records
  conclusions and one that records reasoning.
* **Arguments keep their premises attached** — when you revisit a claim you can
  see *what it was derived from*, so a changed fact invalidates exactly the right
  arguments instead of everything vaguely related.
* **First principles stay baked in** — the axioms that everything else derives
  from can't be silently archived away by an aggressive cap.
* **The user profile is always present** — the model constantly knows who it's
  working for, which is the cheapest possible guard against generic answers.

## The map builds itself (since G1–G3)

Three behaviours that used to need the explicit API now happen in `process_turn`:

* **Rich extraction.** `LLMDistiller` returns the whole map from one turn —
  `principles`, `profile:[{field,text}]`, `arguments:[{claim,premises}]`,
  `perspectives:[{question,stance,text}]`, `updates:[{old,new,kind}]` — and
  `RuleDistiller` catches the explicit forms offline (`Principle: …`,
  `I am a …`, `on one hand … on the other …`, `Actually, …`).
* **Edges on merge.** An argument's premises are committed (deduplicated
  against what is already in memory) and the claim is linked to the *live*
  premise entries.
* **Supersession.** A correction retires the statement it replaces
  (`store.retire`): the successor inherits usage, pin, links and inbound
  edges; the predecessor goes to `archive/` with a `superseded by …` note.
  Matching is on *subject* (`subject_overlap`, content tokens), so "the demo is
  Monday" replaces "the demo is Friday" while "SQLite for tests" leaves
  "Postgres in prod" alone. `engine.supersede(old_id, text)` is the explicit form.

And one at prompt time:

* **Closure under dependencies.** `Context.build` scores entries with a
  `link_weight` term (things other knowledge rests on are worth more) and then
  runs `expand_selection`: a selected argument brings its premises within the
  budget — evicting low-value unlinked leaves to make room, or dropping the
  claim rather than showing it unsupported. Premises render under the claim:

  ```
  ARGUMENTS
  ## Derived claims (⇒ claim, ← premise)
  - ⇒ Forecast skill is bounded no matter how good the model is.
      ← Close trajectories diverge exponentially in a chaotic system.
      ← Initial conditions are only known to finite precision.
  ```

  `MemoryPolicy(link_weight=0, expand_links=False)` turns the map back into a
  list — useful as an ablation.
