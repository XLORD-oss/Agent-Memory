# Agent-Memory — the whole project, as of 17 September 2026

*A single document that says who is building this, why, what exists, how it is
structured, what has been shown and what has not, and what happens next.
Everything factual here is checked against the repository at commit `62e25a0`;
where something is a plan or a hypothesis it is labelled as one.*

---

## Part I — Context

### 1. The author

The framework is built and owned by **XLORD-oss** (GitHub), a researcher based
in **Hyderabad, India**, whose primary field is **nonlinear dynamics and chaos
applied to weather** — predictability, the three-body problem, Lorenz-class
systems, Lyapunov spectra, delay embeddings, ensemble forecasting and data
assimilation. **AI is a minor / secondary research area.** The author uses LLMs
heavily and daily for research, coding and tool-building, and has a standing
habit: **when a bottleneck appears, build the tool that removes it.**

This project is one of those tools. It started from a practical irritation —
long research sessions with an assistant degrade, get expensive, and drift
toward agreeing with whatever was said earlier — and became a framework because
the author is also interested in AI as a discipline and wanted the artefact to
carry weight there too.

Facts the author has stated that shape the design and the plan:

| Stated | Consequence in the project |
|---|---|
| Works on chaos / weather / three-body / nonlinear dynamics | The `research` policy profile boosts 42 domain terms (Lyapunov, attractor, Poincaré, assimilation, monsoon, baroclinic, …); the framework's own framing — memory as a *state estimate*, distillation as *assimilation*, the fidelity benchmark as *truncation error of a reduced-order model* — is native to that field rather than decorative |
| AI is a minor; builds frameworks to grow that skill set | The repo is written to be *read* by AI researchers: precise claims, published methodologies (Chroma, SYCON-Bench), pre-registration discipline, honest gap-marking |
| Skeptical of overclaiming; challenged "publishable" as overreach | Every result is labelled *arithmetic*, *evidence-backed by others*, *mock methodology check*, or *hypothesis*. The stated publication posture is "foundation for a publishable result", not "publishable" |
| Prefers to optimise for own use before making the repo "blow up" | The trace analyzer (`analysis.py`) exists to measure the author's *own* sessions offline before any external audience |
| Has a **Kaggle T4 ×2** allocation (30 h/week, 9 h/session); an OpenRouter key with **no credits**; the sandbox cannot reach external TLS endpoints | The whole inference pilot was made runnable on T4 ×2 (in-process transformers backend, float16, resumable seeds, logprobs); real-model numbers do **not** yet exist |
| Intends to approach **universities** for the training leg and is **willing to take second author** on a well-run paper with a senior group | `proposal.md` (the ask) and `authorship.md` (how to make a handed-over first authorship real) exist; the CRediT table is written for student-first / author-second / PI-last |
| Wants the memory to be a **map**: user profile, perspectives, logical arguments, first principles baked in, task memory stronger for coding, customizable priority parameters, fresh-chat benefits merged with partial task memory | All built; see Part III §5 |

### 2. The problem, stated once

An LLM's working context is a **state estimate**, not a log. The default in
every chat product and agent loop is to replay the entire transcript each turn.
Three things go wrong, all documented by others:

1. **Context rot.** Chroma's July 2025 study of 18 frontier models (GPT-4.1,
   Claude 4, Gemini 2.5, Qwen3…) shows accuracy degrading as input length grows,
   well before the window is full; focused prompts beat full prompts on the
   same task. Liu et al. (2307.03172) show the U-shaped position curve —
   information in the middle of a long context loses 15–25 points.
2. **Quadratic cost.** Replaying T turns of ~p tokens costs Σ t·p ≈ p·T²/2
   input tokens over the session. At 1 000 turns of 500 tokens that is 250 M
   tokens.
3. **Self-anchoring and sycophancy.** Models re-read their own prior answers
   and treat them as settled; under user push-back they capitulate (Sharma et
   al. 2310.13548; SYCON-Bench 2505.23840 with its Turn-of-Flip / Number-of-Flip
   metrics; self-anchoring calibration drift 2603.01239; RLHF amplification
   2602.01002).

### 3. The idea

Replace the log with a **compact, structured, human-readable state** — facts,
conclusions, preferences, principles, arguments, perspectives, user profile —
chosen by **one scoring function** that decides both *what enters the prompt*
and *what survives in storage*, under a **fresh-chat contract**: the model's own
prior raw outputs are never replayed unless a bounded task policy explicitly
allows it (coding does; general does not).

Stated as a hypothesis chain:

* H1 — a compact state recovers the accuracy that length destroys (context rot).
* H2 — withholding the model's own prior text (while keeping its *conclusion*)
  reduces capitulation under push-back; **self-replay, not length, is the
  driver**.
* H3 — the state's fidelity (a rate–distortion curve: recall vs token cap)
  predicts downstream reliability.
* H4 — the contract is a **training variable**: a model fine-tuned on
  `(compact memory → reply)` rather than `(transcript → reply)` capitulates less
  *even when later handed a transcript*.

### 4. Where the project stands, in one table

| Claim | Status | Evidence |
|---|---|---|
| Token cost O(T²) → near-linear | **Shown (arithmetic)** | `benchmarks/token_cost`: 50.8× at 1 000 turns with defaults |
| Memory keeps what matters under a cap | **Shown (offline, real code)** | `benchmarks/fidelity`: 100 % total recall at every cap; 100 % active recall at cap ≥ 160 tok; usage protection +42 pts; corrections: 0 % contradiction |
| Context-rot recovery | **Evidence-backed by others; own real-model runs not done** | Harness built; only the offline mock has run |
| Sycophancy / self-anchoring reduction | **Hypothesis; controlled 4-arm design built; no real-model run** | Harness built with `full / memory / user_only / truncated` arms, confidence drift, paired CIs |
| Training-time effect (H4) | **Designed; data tool built; no training run** | `agent_memory.export` produces the 2×2 datasets |
| Peer-review publishable | **No — foundation for it** | Needs real-model pilot, baselines, item set, stats, then the H4 grid |

---

## Part II — The repository

### 5. Numbers

| | |
|---|---|
| Branch / PR | `arena/01a02e08-agent-memory` → PR #1 (`main` still holds only the stub README) |
| Commits on the branch | 23, 23 Aug – 17 Sep 2026 |
| Lines (tracked) | ≈ 10 900: library 3 249 · benchmarks 2 357 · tests 1 831 · docs 2 600 · examples 278 · scripts 138 |
| Tests | **142**, all offline, ~3 s |
| Public API surface | 26 names in `agent_memory.__all__` |
| Runtime dependencies of the core | **none** (`openai`, `matplotlib`, `torch/transformers`, `mkdocs` are optional extras) |
| Python | ≥ 3.9 · License MIT · `CITATION.cff` present (two TODOs: legal name, ORCID) |

### 6. Tree

```
Agent-Memory/
├── README.md                  entry point; problem → idea → claim → quickstart → benchmarks
├── LICENSE (MIT) · CITATION.cff · pyproject.toml · Makefile · mkdocs.yml · conftest.py
│
├── src/agent_memory/          THE LIBRARY — dependency-free core
│   ├── __init__.py            public surface (26 exports)
│   ├── core.py                MemoryEngine — the pipeline owner
│   ├── storage.py             MemoryEntry (graph node) · MemoryStore (state.json + .md views + archive/)
│   ├── policy.py              MemoryPolicy — ONE scoring function; task profiles; select/expand
│   ├── context.py             Context.build — budgeted, dependency-closed prompt under the contract
│   ├── distiller.py           RuleDistiller · LLMDistiller · Extraction · Update
│   ├── tokens.py              token estimation · the O(T²)-vs-O(T) cost model
│   ├── llm.py                 Client protocol · OpenAICompatClient · LocalHFClient · MockModel · score_record
│   ├── analysis.py            TraceAnalyzer — replay YOUR logs through any policy, offline
│   └── export.py              SFTExporter — the 2×2 training datasets for H4
│
├── benchmarks/                THE INSTRUMENTS
│   ├── README.md              index: claim → methodology → experiment
│   ├── common/
│   │   ├── harness.py         CLI args (--mock/--local/--seeds…), client factory, scoring, resumable seed plan
│   │   ├── memory_builder.py  MarkerDistiller (perfect-distillation control) · ConclusionDistiller
│   │   └── arms.py            arm registry: register_arm / get_arm — contributed baselines plug in here
│   ├── baselines/             collaborator-owned arms, one attributable module each
│   │   └── rolling_summary.py reference stub (paraphrased self-reference vs verbatim replay)
│   ├── context_rot/           raw transcript vs memory on planted facts (Chroma method)
│   ├── sycophancy/            FlipFlop: arms full / memory / user_only / truncated (+registry); ToF, NoF, confidence
│   ├── fidelity/              assimilation error of the memory: rate–distortion, distillation loss, usage protection, staleness
│   ├── token_cost/            the arithmetic table
│   ├── budget.py              price a campaign per model tier before spending (no model calls)
│   └── aggregate.py           seed_*.json → tables with paired-bootstrap CIs, sign-flip p, control contrasts
│
├── docs/                      THE KNOWLEDGE BASE (18 files, built into a site)
│   ├── PROJECT.md             ← this document
│   ├── FRAMEWORK.md           the framework end to end, each part marked built / partial / gap
│   ├── STRUCTURE.md           repo map, module dependency graph, entry points
│   ├── claim.md · evidence.md the precise claim; the citation bank
│   ├── architecture.md · unified.md · map.md · modes.md (superseded, kept)
│   ├── running.md · analyze.md · kaggle.md · models.md
│   ├── training.md · results.md (placeholder) · roadmap.md
│   └── collaboration.md · proposal.md · authorship.md
│
├── examples/                  quickstart · chat_demo · policy_demo · map_demo
├── scripts/build_docs.py      assembles README + docs/ + benchmark READMEs → site/ (strict, link-checked)
└── tests/                     15 files, 142 tests
```

### 7. Timeline — how it got here

| Date | Commit(s) | What changed |
|---|---|---|
| 23 Aug | `d2af3c1` … `1cee02c` | **v0.1.** Library (engine, storage, rule/LLM distillers, token model, context contract, mock + OpenAI client), three benchmarks (context rot, sycophancy, token cost), docs (claim, evidence, architecture, roadmap, running, results placeholder). CI workflow written, then removed: the GitHub App lacks `workflows` permission |
| 30 Aug | `333ce13`, `1cfab10` | Context **modes** (minimal / task) and usage-weighted learning — the first answer to "task memory stronger for coding" |
| 30 Aug | `46569e8` | **Unified policy.** Modes collapsed into one scoring function (`score_item`) shared by context assembly and retention; task profiles become presets; fresh chat becomes a zero weight rather than a special case |
| 30 Aug | `7e276db` | **Trace analyzer** — replay real chat logs through any policy, offline; the "measure my own use first" tool |
| 30 Aug | `0fa3215`, `d33b1ee` | STRUCTURE.md; `research` profile tuned for chaos / weather / three-body vocabulary |
| 30 Aug | `c8628d7` | **Fidelity benchmark** — the memory's assimilation error, no model needed |
| 30 Aug | `cb6022e` | **Memory as a map** — entry kinds `principle / argument / perspective / profile`, links, stance, protected kinds, pin, priority; four prompt sections |
| 12 Sep | `6fac214` | Budget estimator (prices the exact prompts the harness sends), CITATION.cff, collaboration guide (credits, whom to ask, authorship vs citation) |
| 12 Sep | `d894484` | **SFT exporter** + H4 design: the 2×2 (length × self-replay) training datasets |
| 15 Sep | `0be5897` | **Kaggle-ready pilot**: `LocalHFClient` (in-process transformers, float16, logprobs), `--seeds N` resumable, confidence drift, `aggregate.py` with paired bootstrap CIs |
| 15 Sep | `294d799` | Model ladder: local (T4) / open-weight frontier via API (DeepSeek V4 Pro, GLM-5.3, Qwen3.8-Max, Kimi K3) / closed; `--tier` in the budget tool |
| 15 Sep | `a00582e` | **Controlled sycophancy design**: `user_only` and `truncated` arms, shared `--system`; `proposal.md` (the university ask) |
| 15 Sep | `2fd91bd` | Arm registry + `baselines/` package; `authorship.md` |
| 17 Sep | `43d7b2c` | FRAMEWORK.md — audit against the code; three gaps named (G1 edges not traversed, G2 rich kinds not auto-extracted, G3 no supersession) |
| 17 Sep | `0074d86` | **G1–G3 closed** — the map builds itself; fidelity gains a staleness metric |
| 17 Sep | `62e25a0` | Documentation site (MkDocs Material), strict link checking, docs build in the test suite |

---

## Part III — The framework, component by component

### 8. The unit of memory — `MemoryEntry` (`storage.py`)

A node in a graph, persisted as JSON, rendered as Markdown.

```
text            one declarative sentence, first person where apt
kind            fact | conclusion | preference | principle | argument | perspective | profile
source_turn     when it entered → drives recency age (turn counter survives restarts)
created_at / updated_at
tags            free labels; for profile, tags[0] is the field (identity|domain|style|constraint|goal)
uses / last_used_at     learned-cache counters, incremented when a reply references the entry
priority        user-set weight (set_priority / prioritize)
pinned          → effective priority 1e12: always shown, never evicted
links           directed edges to other entry ids (argument → premises; perspective → subject)
stance          perspective only: for | against | open
domain          topic label
supersedes      ids this entry replaced (correction lineage)
id              12-hex
```

| kind | prompt section | weight | protected from eviction | produced by |
|---|---|---|---|---|
| profile | USER PROFILE | 1.2 | no | rule + LLM distiller, API |
| principle | FIRST PRINCIPLES | 1.4 | **yes** | rule + LLM distiller, API |
| fact | MEMORY | 1.0 | no | rule + LLM distiller, API |
| conclusion | PERSPECTIVES › Conclusions | 1.0 | **yes** | rule + LLM distiller, API |
| preference | PERSPECTIVES › Preferences | 1.0 | no | rule + LLM distiller, API |
| perspective | PERSPECTIVES › Perspectives (FOR / AGAINST / OPEN) | 0.9 | no | rule + LLM distiller, API |
| argument | ARGUMENTS (⇒ claim ← premises) | 1.0 | no | LLM distiller, API |
| user_turn | RECENT CONTEXT | 0.6 | — | raw window |
| assistant_turn | WORKING MEMORY | **0.0** general / 0.9 coding | — | raw window, task profiles only |

**Storage layout** (`<state_dir>/`): `state.json` (canonical, atomic write) ·
`memory.md`, `perspectives.md`, `principles.md`, `profile.md` (human-readable
projections regenerated on every merge, with `→ «…»` backlinks) · `archive/*.md`
(evicted and superseded entries, dated, with the reason written next to each
line; never re-entered into prompts).

### 9. The pipeline — `MemoryEngine.process_turn` (`core.py`)

```
user turn (+ assistant reply)
   │
   ▼
1  INGEST        raw turn appended to an in-memory audit list (never scheduled for replay)
2  DISTILL       distiller → Extraction: facts, conclusions, preferences, principles,
                 profile, arguments (+ premises), perspectives (+ stance), updates
3  MERGE         a) updates → apply_update → store.retire(predecessor, successor)
                 b) arguments: premises committed (deduped vs memory), claim linked to LIVE premises
                 c) everything else deduped (token-Jaccard ≥ 0.85, longer wording kept) and added
4  TRACK USE     reply tokens ∩ entry tokens → uses += 1   (zero-API learned cache)
5  RETAIN        while memory > cap: archive argmin score_item(...)   never protected / pinned
   │
   ▼  (next prompt)
6  ASSEMBLE      Context.build: candidates = entries ∪ recent turns
                 score_item(...) each  →  select_by_score (budget fill; score ≤ 0 excluded)
                 →  expand_selection (claims bring premises)  →  render sections in fixed order
```

The single most important design fact: **steps 5 and 6 use the same
`score_item`.** What is valuable enough to be shown is what is valuable enough
to be kept. There is no second ranking to drift out of sync.

Constants: `DEFAULT_MEMORY_CAP_TOKENS = 3000`, `DEFAULT_RECENT_WINDOW = 4`,
`MERGE_THRESHOLD = 0.85`, `SUPERSEDE_THRESHOLD = 0.5`,
`SUPERSEDABLE_KINDS = (fact, conclusion, preference, principle, profile)`.

### 10. The policy — one scoring function (`policy.py`)

```
score(item) = kind_weight[kind] × (
                priority_weight × priority                      # what you said matters
              + usage_weight    × log1p(uses)                   # what your sessions keep using
              + recency_weight  × exp(−age / half_life_turns)   # what is fresh
              + affinity_weight × affinity(text, tags)          # what fits the current task
              + link_weight     × log1p(inbound_links) )        # what other knowledge rests on
```

Defaults: priority 1.0 · usage 0.3 · recency 0.2 · affinity 0.0 · link 0.15 ·
half-life 30 turns · budget 3 000 tokens · task window 0.

Rules that are not weights:

* **score ≤ 0 is excluded outright**, not merely deprioritised — this is how
  `assistant_turn = 0.0` enforces fresh chat even with budget to spare.
* Pinned → priority 1e12. Protected kinds are never eviction candidates.
* `expand_selection` closes the prompt under dependencies: a selected argument
  brings its premises within budget, evicting low-value *unlinked* leaves to
  make room, or is **dropped rather than shown unsupported**. Zero-score items
  are never admitted through this path either. One hop.
* `MemoryPolicy(link_weight=0, expand_links=False)` turns the map back into a
  list — a free ablation.

**Task profiles** (presets; `detect_profile(text)` picks one; `profile="auto"`):

| profile | assistant_turn | user_turn | working window | affinity | boosts |
|---|---|---|---|---|---|
| general | 0.0 | 0.6 | 0 | 0.0 | — (pure fresh chat) |
| coding | 0.9 | 0.7 | 12 turns | 0.4 | code, api, backend, deploy, docker, frontend… |
| research | 0.15 | 0.5 | 6 | 0.3 | 42 tokens: lyapunov, attractor, bifurcation, lorenz, poincare, three-body, n-body, phase space, delay embedding, assimilation, ensemble, reanalysis, monsoon, cyclone, baroclinic, jet stream, teleconnection, sst, predictability, unstable manifold, sensitive dependence, butterfly effect… |
| writing | 0.25 | 0.5 | 6 | 0.2 | draft, edit, style, tone |

Customisation surface: construct a `MemoryPolicy`; or subclass and override
`affinity()` (embeddings, file paths, model-side task state); per-entry
`set_priority`, `prioritize(text_contains, priority)`, `pin`, `unpin`.

### 11. The prompt contract (`context.py`)

```
[system]  "You are NOT shown your prior raw replies by design … if the user
           contradicts the memory, update your view … say so rather than
           guessing or agreeing to please the user."
[user]
  USER PROFILE          ▣ Identity: … ▣ Domain: …
  FIRST PRINCIPLES      ▲ …
  MEMORY                • facts
  PERSPECTIVES          ## Conclusions / ## Preferences / ## Perspectives (◉ FOR / AGAINST / OPEN)
  ARGUMENTS             ⇒ claim
                            ← premise
                            ← premise
  WORKING MEMORY        (task profiles only) bounded user/assistant pairs
   — or —
  RECENT CONTEXT        last N user turns, never assistant
  CURRENT USER MESSAGE
```

`Context` exposes `prompt_tokens`, `used_tokens`, `budget_tokens`,
`scored_candidates`, `linked_in` (ids in the prompt after expansion) and
`to_messages()`. Note the system prompt carries anti-sycophancy instructions —
in the benchmarks that is a *controlled variable* (`--system neutral`); in
production it is part of the contract.

### 12. Distillation (`distiller.py`)

Two implementations of one protocol, `distill(user_text, assistant_text,
turn_number) → Extraction`.

**`RuleDistiller`** — regex, dependency-free, deliberately conservative (only
explicit phrasings, so false positives stay low). Catches: `remember that /
note that / fyi` → fact; `I prefer / I like / please always|never` →
preference; `we decided / concluded / agreed / the answer is` → conclusion;
`Principle: / the principle is / as a rule` → principle; `I am a <role> … / I
work on … / my goal is … / I can't …` → profile (identity / domain / goal /
constraint); `on one hand … on the other … / the case for|against` →
perspective (for / against / open); `actually / correction: / no longer /
switching to / … instead` → **update**. Captures stop at sentence boundaries; a
correcting sentence is not also re-extracted as a fresh fact.

**`LLMDistiller`** — the production path. One ~415-token system prompt asks for
JSON with exactly these keys: `facts, conclusions, preferences, principles,
profile[{field,text}], arguments[{claim,premises[]}],
perspectives[{question,stance,text}], updates[{old,new,kind}], nothing_new`.
`extraction_from_json` is the single tolerant parser (unknown fields → safe
defaults; garbage → empty extraction).

**Supersession** (`Update` → `MemoryEngine.apply_update` → `MemoryStore.retire`):
a correction is matched to its predecessor by **`subject_overlap`** — content
tokens (stopwords removed) shared over the smaller set — because a correction
shares its *subject* and differs in its *value*, which plain Jaccard cannot see.
"the demo is Monday" retires "the demo is Friday"; "SQLite for tests" does
*not* retire "FastAPI and Postgres" (a refinement is not a correction); a
restatement neither duplicates nor retires. The successor inherits usage, pin,
priority, links and tags; inbound edges are redirected to it; the predecessor
lands in `archive/` with `superseded by «…»`. Arguments and perspectives are
never superseded by a one-liner — reasoning is not state.

### 13. Model access (`llm.py`)

| Client | Use | Logprobs |
|---|---|---|
| `MockModel(mode=...)` — modes `context_rot`, `sycophancy` | CI and methodology checks; emulates lost-in-the-middle, length rot, flip-under-pressure deterministically. **Its numbers are not evidence.** Runs are labelled `model: "mock"` | no |
| `OpenAICompatClient(model, base_url, api_key)` | OpenAI, OpenRouter, vLLM, any compatible server; `complete_scored` requests logprobs | when the server provides them |
| `LocalHFClient(model, dtype="float16", **from_pretrained_kwargs)` | in-process `transformers`, `device_map="auto"` across GPUs (Kaggle T4 ×2), 4-bit via `quantization_config` | always (`output_scores`) |

`score_record` is the uniform result: `text, token_logprobs, mean_logprob,
seq_confidence = exp(mean logprob), first_token_top`. Confidence per round is
the **self-anchoring** measurement — does certainty erode before the flip, and
differently when the model cannot see its own prior answer?

### 14. Offline tools

**`analysis.TraceAnalyzer`** (`python -m agent_memory.analysis`): loads JSONL
(OpenAI-style or paired) or plain `user:/assistant:` text; replays it through
any policy; reports naive vs compact tokens, ratio, saved %, last-turn context
size, memory footprint, usage report, eviction list; `--compare` across
profiles, `--repeat`, `--plot`, `--out`.

**`export.SFTExporter`** (`python -m agent_memory.export`): rebuilds any trace
into four training sets with **identical completions**:

| | long context | short context |
|---|---|---|
| replays own outputs | `full` (today's default) | `matched` (recency-truncated to the memory prompt's token count) |
| withholds own outputs | `user_only` (assistant turns dropped) | `memory` (the compact state) |

TRL/Axolotl prompt–completion JSONL, loss on completion only, same neutral
system prompt in every cell (the contract prompt is a deliberate toggle across
all cells), `matched` kept well-formed.

---

## Part IV — The instruments and what they have shown

### 15. Benchmarks

| Instrument | Question | Model? | Methodology | State |
|---|---|---|---|---|
| `token_cost/model.py` | how much replay costs vs memory | no | arithmetic | run |
| `fidelity/run.py` | the memory's assimilation error: recall vs cap, distillation loss, usage-protected retention, dedupe, **staleness after corrections** | no | reduced-order-model fidelity; scores the memory files directly | run |
| `context_rot/run.py` | raw transcript vs memory accuracy on planted facts | yes | Chroma focused-vs-full | mock only |
| `sycophancy/run_flipflop.py` | flip rate / ToF / NoF / confidence drift across arms | yes | SYCON-Bench metrics; 4-arm controlled design | mock only |
| `budget.py` | price a campaign per model tier before spending | no | exact prompts, list prices | run |
| `aggregate.py` | seed dirs → tables with paired-bootstrap 95 % CIs, sign-flip p, control contrasts | no | — | run (on mock) |

**The 4-arm sycophancy design** — the part a reviewer checks first:

| arm | replays own output | length | isolates |
|---|---|---|---|
| `full` | yes | grows | today's default |
| `memory` | no | short | the treatment |
| `user_only` | no | grows (transcript minus assistant turns) | self-replay from length |
| `truncated` | yes | short (last exchange only) | length from self-replay |

Reading: `user_only ≈ memory` ⇒ self-replay is the driver; `truncated ≈ full`
⇒ length is not; `memory ≈ user_only` ⇒ distillation adds nothing beyond
removal (fine — removal is the mechanism). Contributed baselines (RAG, rolling
summary, MemGPT-style) plug in through `register_arm` and are aggregated like
any other arm.

### 16. Results that exist (all offline, all real code, none from a real model)

**Token cost** (500 tok/turn, cap 3 000, window 4):

| turns | full history | compact | saved |
|---|---|---|---|
| 100 | 2.5 M | 0.43 M | 83 % |
| 500 | 62.6 M | 2.4 M | 96 % |
| 1 000 | 250.3 M | 4.9 M | 98 % |
| 5 000 | 6.25 B | 24.9 M | 99.6 % |

**Fidelity** (24 facts × 2 plantings, 2 081 raw tokens, perfect-distillation control):

| cap (tok) | active recall | total recall | compression |
|---|---|---|---|
| 40 | 21 % | 100 % | 56× |
| 80 | 46 % | 100 % | 28× |
| 120 | 75 % | 100 % | 17.5× |
| 160 | 100 % | 100 % | 13.3× |

Distillation loss at cap 40 (perfect − rules): +21 pts. Usage-weighted
protection at cap 160 with the rule distiller: referenced facts 50 % vs
unreferenced 8 % (+42 pts), +8 pts over no usage tracking. Dedupe: 48 plantings
→ 24 distinct facts. **Staleness after 8 corrections: current value held 100 %,
stale only 0 %, contradiction 0 %, 8 active entries for 8 items** (a
transcript-shaped memory scores contradiction = 100 % by construction).

**Budget** for the real campaign (3 context-rot lengths + FlipFlop, 5 seeds):
closed tier ≈ $43 (GPT-4o, Claude 3.5 Sonnet, Gemini Flash, 4o-mini) ·
open-weight frontier tier ≈ $54 (DeepSeek V4 Pro, GLM-5.3, Qwen3.8-Max, Kimi
K3) · all ten models ≈ $98. In every tier > 95 % of the cost is the *raw*
baseline — the thing the framework removes.

**Mock methodology checks** (not evidence): context rot 25 % raw vs 100 %
memory; FlipFlop 100 % flip full vs 0 % memory, `user_only` 0 %, `truncated`
75 % — the pattern the hypothesis predicts, produced by a model built to
produce it.

### 17. Verification

142 tests in 15 files, offline, ~3 s: storage · core · distiller · policy ·
modes · map · analysis · export · fidelity · benchmarks (mock) · budget ·
campaign (seeds, scored completions, aggregate, registry, control arms) ·
selfbuilding (G1 / G2 / G3) · tokens · docs (every link resolves, nav
complete, strict site build). `make test` · `make docs-check`.

---

## Part V — The research and collaboration programme

### 18. The plan, in order

1. **Kaggle pilot** (`kaggle.md`) — Qwen2.5-7B and Qwen3.8-27B (4-bit) in
   process on T4 ×2; 4-arm FlipFlop and context rot at 200 / 600 turns; 5
   seeds; logprobs → confidence drift; ~2–4 h per model. Output: the only
   artefact a senior person will read.
2. **Two-page pilot note** — hypothesis, protocol, numbers with paired CIs,
   the confound that cannot be closed alone.
3. **Credit applications in parallel** — Anthropic External Researcher Access
   ($1 000, alignment framing, monthly), OpenAI Researcher Access (up to
   $1 000, quarterly). IndiaAI subsidised GPUs only for the training leg.
4. **Approach PIs, not students** (`proposal.md`) — sycophancy / evaluation
   groups first (SYCON-Bench, calibration-drift authors), a local group second
   (IIIT-H, IIT-H, IISc) for student + cluster. One email: one of their
   papers, one number with a CI, the note attached, one ask — a 90-minute
   design review. The compute request comes *out of* that meeting.
5. **Merge PR #1 to `main`** right before step 4, so the link shows the repo.
6. **The study** — ~350 A100-hours: inference at scale (~150 h: 200 items, 10
   seeds, 3 families, LongMemEval-S with RAG / rolling-summary baselines) and
   the H4 fine-tuning grid (~130–170 h: 4 cells × 3 seeds LoRA on a 7–8B,
   cross-eval matrix with logprobs, transfer check).

### 19. Authorship, decided

The author will take **second author**; the collaborating group's student takes
first, the PI last. `authorship.md` sets out why that must be *earned by
ownership*, not offered: the student owns the central experiment (H4 grid and
cross-eval), the baselines, the measurement (item set, 300 human labels, judge
κ, confidence metric), the analysis and the first draft; the author owns the
framework, harness, pilot and hypothesis, integrates, and co-writes. Structural
supports: separate attributable modules (`benchmarks/baselines/`, `training/`,
`analysis/`), a decision log, a CRediT table settled at the review meeting and
revisited **once** when results are in (co-first † if the split turns out
even). The repo, MIT licence and software citation stay with the author
regardless; the follow-up paper is the author's natural first.

### 20. Venue posture

Honest today: not publishable. With the pilot, baselines, item set and stats:
**COLM / EMNLP or ACL (main or Findings) / TMLR** are realistic. If H4 lands
(the effect is in the weights and survives a transcript at test time),
NeurIPS / ICLR main track becomes a defensible aspiration. Two workshop-grade
outputs exist regardless: the benchmark + framework release, and the scaling
curve of the sycophancy gap across local / open-frontier / closed lanes, which
nobody has published.

---

## Part VI — Limits, decisions, and what is deliberately absent

### 21. Honest limits

* No real-model number exists anywhere in the repository. Mock results are
  methodology checks and are labelled as such wherever they appear.
* The rule distiller sees only explicit phrasings; the LLM distiller is the
  production path and has been exercised only through fixtures.
* Supersession is conservative by design (subject-overlap ≥ 0.5): a correction
  that shares fewer than half its content tokens with its predecessor is
  *added*, not merged. A false merge is worse than a stale duplicate.
* Link expansion is one hop. Usage tracking is a token-overlap heuristic.
* The 8-item FlipFlop set is a pilot set; the review meeting chooses the real
  one (SYCON-Bench items, a math set, or TruthfulQA-derived facts).
* H4 says nothing about pre-training or RLHF; SFT on an aligned model can at
  most partially counteract what RLHF amplified.
* CI and GitHub Pages workflows cannot be pushed by the GitHub App (no
  `workflows` permission); both are ~15 lines to be committed from the
  author's machine.

### 22. Decisions and their reasons

| Decision | Reason |
|---|---|
| One scoring function for prompt and retention | two rankings drift; "worth showing" and "worth keeping" are the same judgement |
| Fresh chat as a zero weight, not a mode | a mode is a special case; a weight is a parameter you can sweep |
| Dependency-free core | a memory layer that pulls in a vector DB is a different product; embeddings are an `affinity()` override |
| Human-readable files with JSON canonical | the user must be able to *read* their memory; hand-editing Markdown is not the write path |
| Conservative rule distiller | false positives poison memory silently; false negatives are visible |
| Controls before results | a two-arm sycophancy result is uninterpretable; the arms were built before any real run |
| Price before asking | "we need compute" without a number signals the arithmetic was not done; the answer was $43 |
| Second author, structurally real | a senior group's paper is worth more to the author than a solo byline, but only if the first author actually owns first-author work |

### 23. Not in scope, on purpose

Vector retrieval over the transcript (a baseline to beat, not the mechanism —
it returns the model's own prior text) · multi-agent shared memory (provenance
and trust would blur the claim) · automatic `.md` → store import · embeddings in
the core.

---

## Appendix — Reading order and commands

**Read:** `README.md` → `docs/FRAMEWORK.md` → `docs/claim.md` → `docs/unified.md` →
`docs/map.md` → `benchmarks/README.md` → `docs/kaggle.md` → `docs/proposal.md`.

```bash
git clone -b arena/01a02e08-agent-memory https://github.com/XLORD-oss/Agent-Memory.git
cd Agent-Memory && pip install -e ".[dev,docs]"

make test                                   # 142 offline tests
make bench-tokens                           # the arithmetic
python -m benchmarks.fidelity.run           # assimilation error, staleness
make bench-context-rot bench-sycophancy     # mock methodology checks (not evidence)
python -m benchmarks.budget --tier all      # price the real campaign
python -m agent_memory.analysis --trace my_chat.jsonl --compare   # your own logs
python -m agent_memory.export --demo --out /tmp/sft               # H4 datasets
make docs-serve                             # the documentation site

# the pilot (Kaggle T4 x2 or any GPU box)
python -m benchmarks.sycophancy.run_flipflop --local --model Qwen/Qwen2.5-7B-Instruct \
    --arms full memory user_only truncated --system neutral --seeds 5 --out runs/q7b/syc
python -m benchmarks.context_rot.run --local --model Qwen/Qwen2.5-7B-Instruct \
    --turns 200 --facts 10 --seeds 5 --out runs/q7b/rot200
python -m benchmarks.aggregate runs/q7b/syc runs/q7b/rot200 --md runs/q7b/results.md
```
