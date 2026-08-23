# Architecture

## The core move

Instead of replaying the raw transcript, the agent keeps a small set of curated
files that are the **only** memory the model ever sees:

```
state_dir/
├── memory.md          # compact fact store — grows only when a NEW fact appears
├── perspectives.md    # conclusions & preferences — the distilled "what was decided"
├── state.json         # machine-readable source of truth (atomic writes)
└── archive/           # older entries, split off by era — out of active context
    └── 2026-08.md
```

Every turn runs through a five-stage pipeline:

```
  raw turn ──> INGEST ──> DISTILL ──> MERGE ──> ARCHIVE ──> BUILD CONTEXT ──> model
                 │            │           │          │              │
                 │      candidate facts/   dedupe,   keep active    assemble
                 │      conclusions/       commit    memory under   memory +
                 │      preferences                 the token cap  perspectives +
                 │                                                    recent user turns
                 └── raw log kept for audit, NEVER replayed
```

## The five stages

### 1. Ingest
Each raw turn is appended to an audit log. The log exists for accountability and
debugging — it is deliberately **not** part of the context. (This is the design
difference from full-transcript replay: the transcript is a record, not an input.)

### 2. Distill
The distiller extracts only what is *genuinely new* and *durable*:

* `fact` — durable knowledge ("works at NASA", "deploys every Friday 9am").
* `conclusion` — decisions and settled positions ("concluded X, confirmed").
* `preference` — stable values and style ("prefers dark mode", "never meetings before 10am").

Two implementations ship:
* `RuleDistiller` — deterministic patterns, dependency-free (offline, tests, demos).
* `LLMDistiller` — the production path: a model returns structured JSON
  (facts/conclusions/preferences/nothing_new) from each turn. Nothing-new turns
  commit **zero** tokens to memory — this is what makes growth sub-linear.

### 3. Merge
Candidates are deduplicated against existing memory (token-overlap similarity at
a high threshold, so facts differing in one informative token are *not* merged).
Duplicates refresh the timestamp; new entries are committed. `memory.md` and
`perspectives.md` are re-rendered after every commit, so memory is always
human-readable and auditable.

### 4. Archive
When active memory exceeds the token cap (`memory_cap_tokens`, default 3,000),
the oldest entries — facts and preferences first, conclusions last — are moved
into era-dated `archive/*.md` files. Archiving is what keeps the active context
bounded, which is what makes total token cost **truly linear** rather than just
"quadratic with a small constant".

### 5. Build context (the fresh-chat contract)
The prompt is assembled as:

```
SYSTEM  — the memory contract (see below)
MEMORY          — memory.md
PERSPECTIVES    — perspectives.md
RECENT CONTEXT  — the last K raw USER turns (for conversational flow, K default 4)
CURRENT USER MESSAGE
```

Critically, **prior assistant raw outputs are never included**. The model cannot
anchor on "I already said X", because its raw past words are not there. What it
sees instead is the distilled "concluded X" — the decision without the social
momentum that built up to it. This is the anti-self-anchoring mechanism, and it
is what the sycophancy benchmark measures.

The system prompt makes the contract explicit:
> *You are NOT shown your prior raw replies by design... The memory is maintained
> by a separate process and may have been corrected. If the user contradicts the
> memory, update your view; do not cling to an earlier position.*

## Why the token math works

**Full-history replay:** turn N re-sends turns 1..N-1, so over T turns the model
processes `s·T·(T+1)/2` tokens (s = avg tokens/turn). **O(T²).**

**Compact memory:** let `g` = average tokens committed to memory per turn (only
genuinely new facts; `g ≈ s/compression`). Memory at turn N is `min(N·g, cap)`.
Total tokens processed ≈ `Σₙ (min(n·g, cap) + K·s)`, which is:

* **~O(T)** once `N·g` reaches `cap` — i.e., for any conversation longer than
  `cap/g` turns, which is the regime that matters for long-running agents.

Two honest caveats:
* An *unbounded* memory (no archiving) is still O(T²) asymptotically — just with
  a constant `g/s` much smaller than 1. Archiving is what buys true linearity.
* The fixed system prompt is a constant overhead per turn; at very short
  conversations (single digits of turns) replay can look "cheaper". The crossover
  is small, and the saving compounds as T grows. `benchmarks/token_cost/` models
  all of this exactly.

## Design properties (deliberate)

* **Human-readable by design.** Memory is Markdown, not a vector database. A user
  can read, correct, and trust it. This is both a UX property and an auditability
  property.
* **Model-agnostic.** The core is pure Python with zero hard dependencies; the
  LLM client is a thin protocol, so any OpenAI-compatible endpoint plugs in.
* **Testable offline.** `MockModel` emulates the documented failure modes, so the
  whole pipeline and both benchmarks run in CI without an API key.
* **Editable.** Because memory is a file, a human (or a later system) can correct
  a wrong entry — which also means the "memory may have been corrected" contract
  in the system prompt is honest, not aspirational.

## What this is not

* Not a RAG system over transcripts (though retrieval could feed the distiller).
* Not a vector store — compactness comes from curation, not embeddings.
* Not a summarizer — summaries of *everything* still rot; the memory keeps only
  what is durable and new.
