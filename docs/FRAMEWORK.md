# Agent-Memory — the framework, end to end

*A verification document. Every box is marked **[built]**, **[partial]**, or
**[gap]** against the code at the commit in `git log -1`, so you can check it
against the vision rather than against the marketing. Line references are to
`src/agent_memory/` unless stated.*

---

## 0. The claim in one paragraph

An LLM's working context is a *state estimate*, not a log. Full-transcript
replay hands the model a growing, unfiltered log; the model degrades with
length (context rot, lost-in-the-middle), pays O(T²) tokens, and re-reads its
own prior answers under social pressure (self-anchoring, sycophancy).
Agent-Memory replaces the log with a **compact, structured, human-readable
state** — facts, conclusions, preferences, principles, arguments,
perspectives, user profile — selected by **one scoring policy** for both
*what enters the prompt* and *what survives in storage*, under a **fresh-chat
contract**: the model's own prior raw outputs are never replayed unless a
bounded task policy explicitly allows it. The framework ships the
instruments to measure its own effect (fidelity, token cost, context rot,
sycophancy) rather than asserting it.

Status of the claim itself: **token cost — arithmetic, shown. Context rot —
evidence-backed by prior work, our own real-model runs not yet done.
Sycophancy / self-anchoring — a testable hypothesis with a controlled 4-arm
design, not a result.** (`claim.md`, `evidence.md`.)

---

## 1. The vision, item by item

What you said the memory should be, and whether it is:

| Vision item | Status | Where | Notes |
|---|---|---|---|
| Memory like a **map**, not a list — nodes with edges | **[partial]** | `storage.MemoryEntry.links`, `core.link/related`, `map.md` | Graph exists and is persisted/rendered with backlinks. **Edges are not used for selection**: `Context.build` scores nodes independently; a selected argument does not pull in its premises. See gap G1. |
| **User profile** | **[built]** | `kind="profile"`, `profile.md`, `USER PROFILE` prompt section, weight 1.2 | Added via `add_profile_entry`; not auto-extracted (G2). |
| **Different perspectives** (for / against / open) | **[built]** | `kind="perspective"`, `stance`, `PERSPECTIVES` section | Rendered as **FOR / AGAINST / OPEN**. Not auto-extracted (G2). |
| **Logical arguments** (claim + premises) | **[built]** | `kind="argument"`, `add_argument(links=...)`, `ARGUMENTS` section | Premises linked by ID. Not auto-extracted (G2); premises not co-selected (G1). |
| **First principles baked in** | **[built]** | `kind="principle"`, weight 1.4, `PROTECTED_KINDS`, `FIRST PRINCIPLES` section | Never evicted; highest kind weight; pinnable. |
| **Task memory stronger for coding** | **[built]** | `policy.CODING`: `assistant_turn=0.9`, `task_window_turns=12`, `boost_tags={"code",...}` | Bounded working memory of the model's own code/reasoning, coding only. |
| **Customizable priority parameters** | **[built]** | `MemoryPolicy`: `kind_weights`, `priority_weight`, `usage_weight`, `recency_weight`, `affinity_weight`, `half_life_turns`, `budget_tokens`, `task_window_turns`; `set_priority`, `pin` | Every term of the scoring function is a parameter; `affinity()` is overridable. |
| **Fresh-chat + partial task memory merged into one mode** | **[built]** | `unified.md`, `policy.score_item`, `select_by_score` (score ≤ 0 hard-skipped) | One function; profiles are presets. `general` = pure fresh chat. |
| Research-domain awareness (chaos / weather / three-body) | **[built]** | `policy.RESEARCH.boost_tokens` | Lyapunov, attractor, Poincaré, assimilation, ensemble, monsoon, … |
| Memory **learns from use** | **[built]** | `core._track_usage`, `MemoryEntry.uses`, `usage_weight` | Zero-API heuristic: entry counted as used if its tokens appear in the reply. Protects referenced entries from eviction (+42 pts in fidelity bench). |
| **Human-readable**, editable on disk | **[built]** | `memory.md`, `perspectives.md`, `principles.md`, `profile.md`, `archive/` | Markdown with backlinks. |
| Reduces **token cost** | **[built]** (arithmetic) | `tokens.py`, `benchmarks/token_cost` | O(T²) → near-linear; ~50× at 1 000 turns with defaults. |
| Reduces **context rot** | **[built]** (harness), **[unrun]** (real models) | `benchmarks/context_rot` | Chroma-style focused-vs-full. |
| Reduces **sycophancy / self-anchoring** | **[built]** (4-arm harness), **[unrun]** | `benchmarks/sycophancy`, arms `full/memory/user_only/truncated` | The hypothesis. Controls isolate self-replay from length. |
| **Contradictions / updates** ("actually, use SQLite not Postgres") | **[gap]** | — | See G3. Currently a second entry is added; the old conclusion stays. |
| Persists **across sessions** | **[built]** | `MemoryStore` (`state.json` + `.md` projections); turn counter resumes from max `source_turn` | Memory and recency ages survive restart. The raw recent-turn window is in-memory only — by design (fresh chat), each session starts with the map and no transcript. |

### The three functional gaps

**G1 — Edges are stored but not traversed at selection time.**
`Context.build` (context.py:133–146) scores each entry alone. If an
`argument` wins the budget its linked premises may be evicted from the prompt,
so the model sees a conclusion without its support. The map is a map on disk
and a list in the prompt. *Fix:* after `select_by_score`, one hop of link
expansion within the remaining budget (premises of selected arguments; the
subject of selected perspectives), and a `link_weight` term so linked-to
entries score higher. ~40 lines + tests.

**G2 — The rich kinds are not extracted automatically.**
`LLMDistiller` (distiller.py:105–133) and `Extraction` (49–61) know only
`facts / conclusions / preferences`. `principle`, `argument`, `perspective`,
`profile` enter only through the explicit API (`add_principle`, …). In real
use the map will not build itself. *Fix:* extend `DISTILL_SYSTEM` and the JSON
schema to `{facts, conclusions, preferences, principles, profile,
arguments:[{claim, premises:[…]}], perspectives:[{text, stance}]}`; extend
`Extraction`; have `merge` create premise nodes and link them. The rule
distiller can catch a subset (`"the principle is"`, `"on one hand / on the
other"`, `"I am a …"`). ~120 lines + tests.

**G3 — No supersession: a correction adds a sibling instead of retiring the
predecessor.**
`merge` (core.py:97–110) dedupes by token-Jaccard ≥ 0.85; "use SQLite for
tests, not Postgres" and "build with FastAPI and Postgres" overlap below
threshold, so both stay active and the prompt carries a contradiction. This is
the assimilation analogue of never discarding a stale observation. *Fix:* a
`supersedes` field + `retire(entry_id, by=…)`; the LLM distiller returns
`updates:[{old_hint, new}]`; the rule path catches `actually / instead /
no longer / correction:`; retired entries go to the archive with a pointer to
the successor, so the history is kept but never replayed. ~80 lines + tests.

None of the three changes the architecture; they complete it. G2 and G3 are
what make the map *self-building* in real sessions instead of demo-built.

---

## 2. Architecture

```
                 ┌────────────────────────────────────────────────────────────┐
  user turn ───▶ │  1. INGEST      core.ingest        raw turn appended       │
  (+ assistant   │                                    (audit log, in-memory)  │
   reply)        ├────────────────────────────────────────────────────────────┤
                 │  2. DISTILL     distiller.*        turn → Extraction       │
                 │       RuleDistiller (regex, offline)  |  LLMDistiller (JSON)│
                 │       kinds today: fact/conclusion/preference   [G2]       │
                 ├────────────────────────────────────────────────────────────┤
                 │  3. MERGE       core.merge         dedupe (Jaccard ≥ .85), │
                 │                                    keep longer wording,    │
                 │                                    touch()          [G3]   │
                 ├────────────────────────────────────────────────────────────┤
                 │  4. TRACK USE   core._track_usage  reply tokens ∩ entry    │
                 │                                    tokens → uses += 1      │
                 ├────────────────────────────────────────────────────────────┤
                 │  5. RETAIN      core.archive_if_needed                     │
                 │                 while memory > cap: evict argmin score     │
                 │                 (never PROTECTED_KINDS / pinned)           │
                 │                 → archive/*.md                             │
                 ├────────────────────────────────────────────────────────────┤
  next prompt ◀─ │  6. ASSEMBLE    context.Context.build                      │
                 │                 candidates = entries ∪ recent turns        │
                 │                 score_item(...) each  →  select_by_score   │
                 │                 (budget fill, score ≤ 0 excluded)   [G1]   │
                 │                 render sections in fixed order             │
                 └────────────────────────────────────────────────────────────┘
                              ▲                     ▲
                              │                     │
                    MemoryStore (files)      MemoryPolicy (one scoring fn)
                    memory.md perspectives.md    shared by steps 5 and 6
                    principles.md profile.md
                    archive/ memory.json
```

The single most important design fact: **steps 5 and 6 use the same
`score_item`.** What is valuable enough to be shown is what is valuable
enough to be kept. There is no second ranking system to drift out of sync.

---

## 3. The unit: `MemoryEntry` (storage.py)

```
MemoryEntry
  text          declarative sentence, first person where apt
  kind          fact | conclusion | preference | principle | argument | perspective | profile
  source_turn   when it entered (drives recency age)
  created_at / updated_at
  tags          free labels; profile uses tags[0] as the field name; coding boosts {"code",...}
  uses / last_used_at      learned-cache counters
  priority      user-set weight (set_priority / prioritize)
  pinned        → effective_priority = 1e12 (always shown, never evicted)
  links         directed edges to other entry IDs (argument → premises, perspective → subject)
  stance        perspective only: for | against | open
  domain        topic label
  id            12-hex
```

Kinds and their roles in the prompt:

| kind | prompt section | default weight | protected | who creates it today |
|---|---|---|---|---|
| profile | USER PROFILE | 1.2 | no | API only [G2] |
| principle | FIRST PRINCIPLES | 1.4 | **yes** | API only [G2] |
| fact | MEMORY | 1.0 | no | rule + LLM distiller |
| conclusion | PERSPECTIVES › Conclusions | 1.0 | **yes** | rule + LLM distiller |
| preference | PERSPECTIVES › Preferences | 1.0 | no | rule + LLM distiller |
| perspective | PERSPECTIVES › Perspectives (FOR/AGAINST/OPEN) | 0.9 | no | API only [G2] |
| argument | ARGUMENTS (claim ⇒ premises) | 1.0 | no | API only [G2] |
| user_turn | RECENT CONTEXT | 0.6 | — | raw window |
| assistant_turn | WORKING MEMORY | **0.0** (general) / 0.9 (coding) | — | raw window, task profiles only |

---

## 4. The policy: one scoring function (policy.py)

```
score(item) = kind_weight[kind] × (
                priority_weight × priority                      # what you said matters
              + usage_weight    × log1p(uses)                   # what your sessions keep using
              + recency_weight  × exp(−age / half_life_turns)   # what is fresh
              + affinity_weight × affinity(text, tags) )        # what fits the current task
```

* `score ≤ 0` is **excluded outright** (not merely deprioritised) — this is how
  `assistant_turn = 0.0` enforces fresh chat even when budget remains.
* Pinned → priority 1e12. Protected kinds are skipped by eviction regardless
  of score.
* Presets (`TASK_PROFILES`): `general` (fresh chat, no working memory),
  `coding` (assistant 0.9, window 12, code tags boosted), `research`
  (assistant 0.15, window 6, chaos/weather vocabulary boosted), `writing`
  (assistant 0.25, window 6). `detect_profile(text)` picks one from the
  current message; `profile="auto"`.
* Customisation surface: construct a `MemoryPolicy`, or subclass and override
  `affinity()` (embeddings, file paths, model-side task state).

Design principles behind it (`unified.md`): a memory item has *value*; value
is the only currency; the prompt budget and the storage cap are two budgets
spent in the same currency; the fresh-chat contract is a zero weight, not a
special case.

---

## 5. The prompt contract (context.py)

Fixed section order, each present only if non-empty:

```
[system]  SYSTEM_PROMPT — "You are NOT shown your prior raw replies by design …
           if the user contradicts the memory, update your view … say so rather
           than guessing or agreeing to please the user."
[user]
  USER PROFILE          ▣ Researcher: … ▣ Domain: …
  FIRST PRINCIPLES      ▲ …
  MEMORY                • facts
  PERSPECTIVES          ## Conclusions / ## Preferences / ## Perspectives (◉ FOR/AGAINST/OPEN)
  ARGUMENTS             ⇒ claim  → «premise…»
  WORKING MEMORY        (task profiles only) user/assistant pairs, bounded
   — or —
  RECENT CONTEXT        last N user turns, never assistant
  CURRENT USER MESSAGE
```

Note: the system prompt carries anti-sycophancy instructions. In the
benchmarks this is a controlled variable (`--system neutral` runs every arm
without it); in production it is part of the contract.

---

## 6. Storage & persistence (storage.py)

```
<state_dir>/
  state.json         canonical store (all entries, all fields, schema v4; atomic write via .tmp)
  memory.md          facts                              ─┐ human-readable
  perspectives.md    conclusions / preferences / perspectives│ projections,
  principles.md      principles                          │ regenerated on
  profile.md         profile                            ─┘ every merge
  archive/*.md       evicted entries (dated files; never re-entered into prompts)
```

Backlinks render as `→ «first words of the linked entry…»`. Edit the `.md`
files by hand at your own risk — `state.json` is the source of truth; a
future `import_md()` is not built.

---

## 7. Instruments (benchmarks/)

| Instrument | Question | Needs a model? | Status |
|---|---|---|---|
| `token_cost/` | O(T²) vs near-linear — how much | no | built; arithmetic |
| `fidelity/` | assimilation error of the memory: recall vs cap (rate–distortion), distillation loss, usage-protected retention, dedupe | no | built; run |
| `context_rot/` | raw transcript vs memory accuracy on planted facts, Chroma method | yes | built; mock only |
| `sycophancy/` | ToF / NoF / confidence drift across arms `full / memory / user_only / truncated` (+ contributed baselines via `register_arm`) | yes | built; mock only |
| `budget.py` | price the campaign per model tier before spending | no | built |
| `aggregate.py` | seed dirs → paired-bootstrap CIs, sign-flip p, control contrasts | no | built |
| `analysis.py` (library) | replay *your own* logs through any policy offline; naive vs compact tokens, eviction list, policy comparison | no | built |
| `export.py` (library) | 2×2 SFT data (full / user_only / memory / matched) for the H4 training study | no | built; no training run |

Model access: `MockModel` (documented failure modes, CI), `OpenAICompatClient`
(any endpoint; logprobs when served), `LocalHFClient` (in-process
transformers, T4-friendly, logprobs always). `--seeds N` is resumable.

---

## 8. Research programme

| Hypothesis | Instrument | State |
|---|---|---|
| H1 Compact state recovers accuracy lost to context length | context_rot | pilot pending (Kaggle) |
| H2 Withholding own prior outputs reduces capitulation; self-replay, not length, is the driver | sycophancy 4-arm | pilot pending |
| H3 State fidelity (rate–distortion curve) predicts downstream reliability | fidelity × context_rot | fidelity half done |
| H4 The contract is a *training* variable (survives when the trained model is shown a transcript) | export.py + LoRA grid + cross-eval | designed, needs cluster |

Publication posture: foundation for a publishable result; not publishable
as-is (`collaboration.md`, `proposal.md`, `authorship.md`).

---

## 9. What is deliberately *not* in scope (and why)

* **Vector retrieval over the transcript.** It is a baseline to beat
  (`benchmarks/baselines/`), not the mechanism — RAG returns raw text,
  including the model's own prior outputs, which is the thing under test.
* **Multi-agent shared memory.** One user, one map. Sharing adds
  provenance and trust problems that would blur the current claim.
* **Automatic `.md` → store import.** Human-readable is for *reading and
  auditing*; the JSON is canonical.
* **Embeddings in the core.** The core is dependency-free; embeddings are
  an `affinity()` override.

---

## 10. Repository map (for orientation; `STRUCTURE.md` has the full tree)

```
src/agent_memory/   core · storage · policy · context · distiller · tokens · llm · analysis · export
benchmarks/         common (harness, arms registry) · baselines · context_rot · sycophancy · fidelity · token_cost · budget · aggregate
docs/               claim · evidence · architecture · unified · map · analyze · running · kaggle · models
                    training · collaboration · proposal · authorship · results (placeholder) · roadmap · FRAMEWORK (this)
examples/           quickstart · chat_demo · policy_demo · map_demo
tests/              118 offline tests
```
