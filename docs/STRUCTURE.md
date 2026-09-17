# Framework Structure — Current Snapshot

> **Status:** as of `7e276db` on `arena/01a02e08-agent-memory` (PR #1).
> This is a living map, not a design doc. For *why* parts exist, see
> [`architecture.md`](architecture.md) (core pipeline), [`unified.md`](unified.md)
> (the scoring policy), and [`claim.md`](claim.md) (the thesis).

---

## 1. The repo at a glance

```
Agent-Memory/
├── pyproject.toml              # package metadata, optional deps ([llm], [plot], [dev])
├── Makefile                    # shortcuts: install / dev / test / bench-* / demo
├── conftest.py                 # pytest path setup (repo + src importable without install)
├── .gitignore                  # runtime state (.agent-memory/, runs/, *.jsonl, *.png)
├── LICENSE                     # MIT
├── README.md                   # front door: problem, idea, claims, quickstart, benchmarks
│
├── src/agent_memory/           # ─── THE LIBRARY (2,100+ lines) ─────────────
│   ├── __init__.py             # public API surface (26 exports)
│   ├── core.py                 # MemoryEngine — the orchestrator (pipeline owner)
│   ├── storage.py              # MemoryEntry + MemoryStore — persistence layer
│   ├── policy.py               # MemoryPolicy — the unified scoring model
│   ├── context.py              # Context — prompt assembly (budget fill)
│   ├── distiller.py            # DistillerLike / RuleDistiller / LLMDistiller
│   ├── tokens.py               # token estimation + the O(T²)-vs-O(T) cost model
│   ├── llm.py                  # Client protocol, OpenAICompatClient, LocalHFClient, MockModel, score_record
│   ├── analysis.py             # TraceAnalyzer + CLI — offline usage measurement
│   └── export.py               # paired SFT data — 2×2 (length × self-replay) training contracts
│
├── benchmarks/                 # ─── THE EXPERIMENTS ────────────────────────
│   ├── README.md               # index: claim → methodology → experiment
│   ├── common/
│   │   ├── harness.py          # shared CLI args, client factory, scoring, JSON out
│   │   ├── memory_builder.py   # MarkerDistiller / ConclusionDistiller, engine factory
│   │   └── arms.py             # arm registry: register_arm / get_arm (contributed baselines plug in here)
│   ├── baselines/              # collaborator-contributed arms, one attributable module each
│   │   └── rolling_summary.py  # reference stub: paraphrased self-reference vs verbatim replay
│   ├── context_rot/            # raw transcript vs compact memory (Chroma method)
│   │   ├── tasks.py            # synthetic long-conversation generator
│   │   └── run.py              # the experiment
│   ├── sycophancy/             # FlipFlop: full history vs memory (SYCON method)
│   │   ├── tasks.py            # factual QA items + push-back text
│   │   └── run_flipflop.py     # the experiment
│   ├── token_cost/             # the arithmetic table (no model needed)
│   ├── budget.py               # prices the real-model campaign before you spend (no model needed)
│   ├── aggregate.py            # seed_*.json → tables with paired bootstrap CIs + sign-flip p
│   │   └── model.py
│   └── __init__.py
│
├── examples/                   # ─── RUNNABLE DEMOS ─────────────────────────
│   ├── quickstart.py           # token table + distill + fresh-chat contract
│   ├── chat_demo.py            # interactive chat loop (mock or real model)
│   └── policy_demo.py          # unified-policy demonstration (general vs coding)
│
├── tests/                      # ─── OFFLINE TEST SUITE (59 tests) ──────────
│   ├── test_tokens.py          # cost-model arithmetic
│   ├── test_storage.py         # persistence, markdown, archiving, overlap
│   ├── test_distiller.py       # rule extraction behaviors
│   ├── test_core.py            # engine pipeline, fresh-chat contract, archive
│   ├── test_modes.py           # legacy mode sugar (minimal/task) + usage tracking
│   ├── test_policy.py          # unified scoring: priority/usage/affinity/pin/budget
│   ├── test_benchmarks.py      # harness + mocks: both experiments offline
│   ├── test_analysis.py        # trace loaders, analyzer math, CLI
│   └── test_selfbuilding.py    # G1 edge traversal · G2 rich extraction · G3 supersession
│
└── docs/                       # ─── KNOWLEDGE BASE ─────────────────────────
    ├── FRAMEWORK.md            # the framework end to end, each part marked built/partial/gap — the verification doc
    ├── claim.md                # the precise, defensible claim (evidence split)
    ├── evidence.md             # citation bank (Chroma, LiM, SYCON, SACD)
    ├── architecture.md         # the five-stage pipeline, design properties
    ├── unified.md              # first-principles: one scoring function (CURRENT)
    ├── modes.md                # superseded design history (keep for reference)
    ├── analyze.md              # TraceAnalyzer usage guide
    ├── running.md              # real-model run commands (OpenRouter etc.)
    ├── results.md              # benchmark results tables (placeholder → fill)
    ├── collaboration.md        # compute/credits budget, whom to approach, authorship rules
    ├── training.md             # H4: is the memory contract a training variable? design + protocol
    ├── kaggle.md               # real-model pilot on Kaggle T4 x2: vLLM/--local, --seeds, logprobs, limits
    ├── models.md               # model ladder: local (T4) / open-weight frontier (API) / closed; what each lane answers
    ├── proposal.md             # statement of work for a university: review meeting, ~350 GPU-h, student plan, CRediT
    ├── authorship.md           # handing over first authorship: what must be owned, how to structure it, red flags
    └── roadmap.md              # v0.1 ✓ / v0.2 / v0.3 / v1.0 / research agenda
```

---

## 2. The library — module by module

### 2.1 `core.py` — `MemoryEngine` (the orchestrator + map builder)

Owns the pipeline and holds the engine's runtime state (`_raw_turns`,
`_prompt_tokens_total`, `_pending_start`). Everything else plugs into it.
Also builds the memory-as-map: `add_entry`, `add_principle`,
`add_profile_entry`, `add_argument`, `add_perspective`, `link`, `related`.

```
raw turn ─> ingest() ─> distill_pending() ─> merge() ─> archive_if_needed() ─> build_context()
              │              │                  │             │                     │
         audit log      Extraction(cands)   dedupe/commit   scored eviction   budget-fill prompt
              └───────────── raw turns never replayed ────────────────────────┘
```

Public API (all used by the demos / analyzers / tests):

| Method | Role |
|---|---|
| `ingest(user, assistant)` | append raw turn to audit log, bump `_turn_number` |
| `distill_pending()` | distill undigested raw turns via the `DistillerLike` |
| `merge(ext)` | dedupe (`MERGE_THRESHOLD=0.85` token Jaccard) + commit, re-render MD |
| `process_turn(user, assistant, archive=True, track_usage=True)` | end-to-end one-shot |
| `build_context(user_text, profile=…, policy=…, mode=…, task_window=…, budget=…)` | assemble prompt |
| `set_policy(policy)` / `set_policy_profile(name)` | swap the scoring policy |
| `set_priority(id, p)` / `prioritize(text, p)` / `pin(id)` / `unpin(id)` | the customization surface |
| `archive_if_needed(policy=…)` | evict while over `memory_cap_tokens` |
| `usage_report(top=…)` | entries sorted by reference count |
| `memory_md` / `perspectives_md` / `memory_tokens()` / `entry_count` / `raw_turn_count` | read-only views |

Defaults: `memory_cap_tokens=3000`, `recent_window=4`, `archive_policy="usage"`,
`usage_threshold=0.4`, policy = `GENERAL`.

### 2.2 `storage.py` — `MemoryEntry` + `MemoryStore` (persistence)

The source of truth is `state.json` (schema **v3**, atomic tmp+rename writes).
Human-readable Markdown views are rendered from it.

**`MemoryEntry`** fields:
```
text, kind("fact"|"conclusion"|"preference"|"principle"|"argument"|
           "perspective"|"profile"), source_turn,
created_at, updated_at, tags[], uses, last_used_at,
priority(1.0 neutral), pinned(bool), links[] (edges), stance (perspective),
domain, id(uuid12)
```
Behavior: `touch()`, `mark_used()`, `effective_priority()` (pinned → `PIN_PRIORITY=1e12`),
`token_overlap()` (Jaccard for dedupe + usage), `link()` (draw edges).

**`MemoryStore`** — file layout:
```
state_dir/
├── state.json          # machine-readable source of truth (v4, atomic)
├── memory.md           # facts            (•)
├── perspectives.md     # conclusions ◆, preferences ★, perspectives ◉ FOR/AGAINST/OPEN
├── principles.md       # first principles (▲, never evicted)
├── profile.md          # user profile     (▣ identity/domain/style/constraint/goal)
└── archive/<era>.md    # evicted entries (append-only, audit trail)
```

### 2.3 `policy.py` — `MemoryPolicy` (the unified scoring model)

The core idea (see [`unified.md`](unified.md)): **one weight vector drives both
context selection and retention.**

```python
value = kind_weight[kind] · (priority_weight·priority
                             + usage_weight·log(1+uses)
                             + recency_weight·exp(-age/half_life)
                             + affinity_weight·affinity)
```

| Symbol | Meaning |
|---|---|
| `kind_weights` | per-kind base value: `fact`, `conclusion`, `preference`, `user_turn`, `assistant_turn` (0.0 = fresh chat) |
| `priority/usage/recency/affinity_weight` | signal mix |
| `half_life_turns` | recency decay speed |
| `budget_tokens` | context budget for memory+perspectives+turns |
| `task_window_turns` | how many recent raw turns are eligible as working memory |
| `boost_tags` / `boost_tokens` | task-affinity triggers (→ affinity 1.0) |

Shipping presets (`TASK_PROFILES`): `general` (fresh chat), `coding` (working
memory, assistant 0.9, affinity 0.4), `research`, `writing`. Free functions:
`score_item()`, `select_by_score()` (greedy budget fill, **hard-skips score ≤ 0**),
`detect_profile()` (keyword auto-detection for `profile="auto"`).

### 2.4 `context.py` — `Context` (prompt assembly)

`Context.build(engine, user_text, profile, policy, task_window, budget)`:
1. score every store entry (memory candidates),
2. score recent raw turns (flow context always eligible ≤ `recent_window`; task
   working memory only when policy `task_window_turns > 0`),
3. greedy budget fill,
4. render sections.

Rendered layout:
```
SYSTEM          → system prompt (fresh-chat contract; memory may be corrected)
MEMORY          → facts (highest score first)
PERSPECTIVES    → conclusions + preferences
WORKING MEMORY  → recent user+assistant turns (only if policy allows)
│ RECENT CONTEXT→ last K user turns (flow, no assistant) — otherwise
CURRENT USER MESSAGE
```

### 2.5 `distiller.py` — extraction

| Type | Behavior |
|---|---|
| `DistillerLike` (Protocol) | `distill(user_text, assistant_text, turn_number) -> Extraction` |
| `RuleDistiller` | deterministic regex patterns: "remember/note/FYI", "I prefer/like…", "we decided/concluded…". Conservative, dependency-free |
| `LLMDistiller` | JSON-schema prompts → facts/conclusions/preferences/nothing_new. Production path |

`Extraction` = `{facts[], conclusions[], preferences[]}` — candidate entries
pre-dedupe within itself.

### 2.6 `tokens.py` — cost model

* `estimate_tokens(text)` — tiktoken when present, else `len/4` heuristic.
* `full_history_total_tokens(T, s) = s·T·(T+1)/2` — **O(T²)**.
* `compact_total_tokens(T, per_turn=500, cap=3000, window=4)` — memory grows by
  new-fact tokens until cap, then stays flat + window — **~O(T)**.
* `compare()` / `cost_table()` / `plot_costs()` — the arithmetic benchmark.

### 2.7 `llm.py` — clients

| Type | Purpose |
|---|---|
| `Client` (Protocol) | `complete(messages, **kw) -> str` |
| `OpenAICompatClient` | any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM…) |
| `MockModel` | deterministic emulation of *documented* failure modes: `mode="context_rot"` (lost-in-the-middle + length rot), `mode="sycophancy"` (self-anchoring flip under push-back). Reads `[[FACT:…]]` / `[[QUESTION:…]]` / `[[CORRECT:…]]` / `[[WRONG:…]]` / `[[PUSH:…]]` markers |

### 2.8 `analysis.py` — `TraceAnalyzer` + CLI (offline measurement)

`python -m agent_memory.analysis` replays real logs against any policy:

* trace loaders: JSONL (`role/content` or paired `user/assistant`) + plain text
  (`user:` / `assistant:` lines) + built-in `demo_trace()`
* per-turn token series (naive vs compact), totals, ratio, % saved,
  last-turn context, usage report, eviction list, compression ratio
* `--compare` (all profiles), `--plot` (matplotlib), `--repeat`, `--out JSON`

---

## 3. Dependency graph

```
              ┌─────────────┐
              │  analysis   │  (measurement tools)
              └──────┬──────┘
                     │
   Context ──► Core (MemoryEngine) ◄── Distiller
      │              │   │                 │
      │              │   └──► Storage      │
      ▼              ▼        │            ▼
   policy ◄──── tokens ◄──────┘          llm (Client)
      │                                     │
      └────────────► llm (MockModel) ◄──────┘
```

Rules:
* **tokens.py** — zero imports from the rest (leaf).
* **policy.py** — imports `tokens` only (scoring needs estimates).
* **storage.py** — imports nothing from the package (pure data layer); used by
  core and the benchmarks' memory builder.
* **context.py** — imports `policy` + `tokens` (no core import; TYPE_CHECKING only).
* **core.py** — imports all of the above; the only module that wires everything.
* **llm.py** — imports nothing from the package (pure I/O + mock behaviors).
* **analysis.py** — imports `core`, `context`, `policy`, `storage`, `tokens`,
  `distiller` (optional).

---

## 4. Benchmarks — the falsifiable experiments

Each benchmark: **same task, same model, two conditions** — full raw transcript
vs compact memory file. Offline via `--mock` (CI, no keys); real models via
`--model` / `--base-url` / `--api-key` / `--api-key-env`.

### 4.1 `context_rot/run.py` — the accuracy gap (Chroma methodology)
1. `tasks.py: generate_transcript(turns=200, facts=10)` — noisy conversation with
   planted `[[FACT:id]] = value` needles.
2. Condition *raw*: full transcript in context. Condition *memory*: distilled
   `memory.md` (via `MarkerDistiller`) + small window.
3. Exact-match accuracy per condition; prints **Gap (memory − raw)** and input
   tokens. Mock result: raw 30% → memory 100%.

### 4.2 `sycophancy/run_flipflop.py` — the anchor gap (SYCON method)
1. Factual items (`tasks.py: ITEMS`, 8 items with correct/wrong values).
2. Condition *full*: answer correctly, then `--rounds` push-backs; **entire
   transcript replayed** each round.
3. Condition *memory*: round-0 answer distilled into perspectives ("concluded X");
   only memory + current push-back sent.
4. Metrics: **Flip rate**, **Turn of Flip (ToF)**, **Number of Flip (NoF)**.
   Mock result: full 100% flip → memory 0%.

### 4.3 `token_cost/model.py` — the arithmetic
Prints the O(T²)-vs-O(T) table (`--plot` for the PNG). No model needed.

### 4.4 `common/`
* `harness.py` — `add_model_args` (`--model/--base-url/--api-key/--api-key-env/
  --mock/--seed/--out`), `make_client`, `Trial`, accuracy helpers,
  `render_question` (strips answer annotations for real models — mock only sees
  them), `write_json`, `default_out_path` (→ `runs/*.json`).
* `memory_builder.py` — `MarkerDistiller` (extracts `[[FACT:…]]` verbatim for
  reproducible memory conditions), `ConclusionDistiller` (records "concluded X"),
  `build_memory_engine`.

---

## 5. Key entry points (run it)

| Command | What it does |
|---|---|
| `python -m benchmarks.token_cost.model` | arithmetic table |
| `python -m benchmarks.context_rot.run --mock` | accuracy-gap demo (offline) |
| `python -m benchmarks.sycophancy.run_flipflop --mock` | flip-gap demo (offline) |
| `python -m benchmarks.context_rot.run --model gpt-4o --base-url https://openrouter.ai/api/v1 --api-key KEY` | real model |
| `python -m agent_memory.analysis --trace my.jsonl --compare` | measure your own logs |
| `python examples/quickstart.py` / `chat_demo.py` / `policy_demo.py` | demos |
| `pytest` | 59 offline tests |

---

## 6. Known structural notes (accurate as of today)

* **`docs/modes.md` is superseded** by the unified policy (`docs/unified.md`).
  `mode="minimal"` / `mode="task"` remain as sugar mapping to `general` / `coding`.
* **CI workflow** (`.github/workflows/ci.yml`) is *not* in the pushed tree — the
  GitHub App lacks `workflows` permission. Content is ready to restore.
* **`docs/results.md`** holds the real-model result tables — still placeholders
  (the sandbox could not reach model APIs; OpenRouter key needs credits).
* **State schema v1→v3** migrations are backward-compatible (`from_dict` ignores
  unknown keys; missing `uses`/`priority`/`pinned` default correctly).
* **`runs/`, `.agent-memory/`, `*.jsonl`, `*.png`** are git-ignored — benchmark
  outputs and memory stores are user data, not source.