# Modes — development history

> **Superseded.** The modes described here are now **presets of one unified
> scoring policy** — see [`docs/unified.md`](unified.md) for the first-principles
> design. `mode="minimal"` ↔ `general`, `mode="task"` ↔ `coding` remain as sugar;
> everything else is a weight on the vector.

The unification: both axes below are the **same scoring function** applied to
selection (context) and retention (memory). These pages record the original
design of each axis for reference.

One mode for every situation is the design sin this framework avoids. There are
two independent axes, and conflating them is where naive "context optimization"
goes wrong:

* **Presentation modes** — what goes *in* the context each turn.
* **Learning modes** — how the memory store *updates* over time.

The second axis is the one that actually realizes "optimize what the model
learns": it decides which facts earn their tokens.

---

## Axis 1 — Presentation modes (`build_context(mode=...)`)

### `minimal` (default) — the fresh-chat contract

```
SYSTEM
MEMORY          — distilled facts
PERSPECTIVES    — conclusions & preferences
RECENT CONTEXT  — last K raw USER turns only
CURRENT USER MESSAGE
```

Prior assistant outputs are never replayed. The model cannot anchor on "I
already said X". This is the anti-context-rot / anti-self-anchoring default and
the right choice for most conversational work.

### `task` — bounded working memory

```
SYSTEM
MEMORY
PERSPECTIVES
WORKING MEMORY  — last K raw turns VERBATIM (user + assistant), clearly delimited
CURRENT USER MESSAGE
```

The **one deliberate exception** to the fresh-chat rule. Multi-step code,
derivations, and long edits need working-memory precision: the model must see
the exact lines it just wrote to continue coherently. The window is **bounded**
(default 8 turns), so it cannot rot into a full replay — the working memory is
scoped to the active task, not the whole conversation.

**Rule of thumb:** conversational / advisory / research synthesis → `minimal`.
Active construction (code, math, editing a document) → `task`.

### Future: `segmented` (deep work)
A larger window with explicit per-segment labels (`[section: schema change]`,
`[section: results]`) so long derivations stay navigable while rot stays
contained per segment. Designed, not yet built.

---

## Axis 2 — Learning modes (how memory updates)

### `passive` — explicit distillation only
Extract only what the user states as durable ("remember", "I prefer", "we
decided"). Simple, cheap, conservative. This is what ships today as the default
distiller behavior.

### `usage-weighted` — memory as a learned cache  *(implemented)*
The framework's answer to "optimize what the model learns": **track which
memory entries the model's answers actually reference**, and let eviction favor
the unused.

* Zero API cost — pure token-overlap arithmetic between the answer and each
  entry's text (`_track_usage`). No extra model calls, no embedding server.
* An entry referenced in every session is **protected from eviction even when
  it is old**. Dead entries — never referenced — are archived first,
  regardless of age.
* Decisions (conclusions) are never evicted; "concluded X" is the record, not
  the social momentum behind it.
* `engine.usage_report()` shows the memory's real footprint: which entries earn
  their tokens in *your* usage.

This is the mode that matters for long-lived personal/research agents: the
memory converges to what you actually ask about, not what happened to be said
loudly once.

### Future: `active` (implicit learning)
Scan each turn for *implicit* durable facts and user corrections ("no, actually
it's..." → update memory instead of the model re-anchoring). Costs extra model
calls, so it should be batched/async rather than per-turn. This is where
"learning" becomes proactive — designed, not yet built.

---

## The mode matrix

| Need | Presentation | Learning |
|---|---|---|
| Everyday chat, advice, synthesis | `minimal` | `usage-weighted` |
| Code / derivations / edits | `task` | `usage-weighted` |
| Explicit fact capture | `minimal` | `passive` |
| Auto-correct & implicit facts (future) | `minimal` | `active` |

Every mode still keeps the anti-rot core: nothing is ever the full raw
transcript, and the fresh-chat contract holds everywhere the task doesn't
explicitly need working memory.