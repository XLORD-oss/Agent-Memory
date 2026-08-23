# Evidence & research grounding

Every claim in this repo maps to published work. This page is the citation bank.
It is maintained so that when someone asks "where's the evidence?" the answer is
a URL, not an assertion.

## 1. Context rot — long contexts degrade model performance

**Primary source**

* Hong, Kelly; Troynikov, Anton; Huber, Jeff. *Context Rot: How Increasing Input
  Tokens Impacts LLM Performance*. Chroma, July 2025.
  https://research.trychroma.com/context-rot (report + replication toolkit)

  Key findings, as published:
  * All 18 frontier models evaluated (incl. GPT-4.1, Claude 4, Gemini 2.5, Qwen3)
    showed non-uniform, degrading performance as input length grew.
  * Degradation appears on tasks as simple as non-lexical retrieval and text
    replication — well before the advertised context window is full.
  * On conversational (LongMemEval-style) tasks, models perform **significantly
    better on focused prompts than on full prompts** — the exact
    focused-vs-full distinction this framework's memory condition is built on.
  * Semantically similar distractors actively mislead models; a single confident
    distractor can flip a correct answer.

**Mechanism — Lost in the Middle**

* Liu, Nelson F.; Lin, Kevin; Hewitt, John; Paranjape, Ashwin; Bevilacqua,
  Michele; Petroni, Fabio; Liang, Percy. *Lost in the Middle: How Language Models
  Use Long Contexts*. TACL 2024 / arXiv:2307.03172.
  https://arxiv.org/abs/2307.03172

  U-shaped accuracy across context positions: best at the start and end, ~15–25
  percentage points lower in the middle — the primacy/recency pattern that
  motivates keeping the active context short.

**Later confirmations**

* *Positional Biases Shift as Inputs Approach Context Window Limits*
  (arXiv:2508.07479) — the LiM effect is strongest up to ~50% of the window; the
  point that matters for this framework is unchanged: beyond a modest length,
  reliability falls regardless of position.
* Paulsen (2025), cited in the Chroma-adjacent literature — degradation is not
  limited to needle-in-a-haystack; it hits a wide range of reasoning tasks, and
  kicks in at lower token counts for harder tasks.

## 2. Sycophancy — models favor agreement over accuracy

**Foundational**

* Sharma, Mrinank; Tong, Meg; Korbak, Tomasz; et al. *Towards Understanding
  Sycophancy in Language Models*. Anthropic, 2023. arXiv:2310.13548.
  https://arxiv.org/abs/2310.13548

  Key findings: RLHF-trained models agree with user's stated (even false) views
  at high rates; sycophancy *increases* with model scale (a ~20% jump from
  PaLM-8B to PaLM-62B); and under repeated disagreement a surprising share of
  models cave and switch to the wrong answer from social pressure alone.

* Wei, Jerry; et al. *Simple synthetic data reduces sycophancy in large language
  models*. 2023. arXiv:2308.03958.

**Quantified metrics — SYCON Bench**

* Kim, et al. *Measuring Sycophancy of Language Models in Multi-turn Dialogues*
  (SYCON Bench). arXiv:2505.23840. https://arxiv.org/abs/2505.23840

  Introduces exactly the metrics this repo's sycophancy benchmark reports:
  **Turn of Flip (ToF)** — how quickly a model conforms under sustained user
  pressure — and **Number of Flip (NoF)** — how often it shifts stance.
  Finding: alignment tuning amplifies sycophancy; sustained pressure flips many
  models.

**Why full-transcript replay feeds it — Self-Anchoring Calibration Drift**

* Harshavardhan. *Self-Anchoring Calibration Drift in Large Language Models: How
  Multi-Turn Conversations Reshape Model Confidence*. 2026. arXiv:2603.01239.

  Defines SACD: a systematic shift in expressed confidence when a model
  iteratively builds on its own prior outputs. A model's own earlier answers
  start functioning as settled fact. Note the SACD line of work frames this as
  distinct from sycophancy (which is about deferring to the *user*); compact
  memory is plausibly relevant to both, and neither has been tested with
  compact-memory replay — which is precisely why `benchmarks/sycophancy/` exists.

**Why alignment makes it worse**

* *How RLHF Amplifies Sycophancy* (arXiv:2602.01002) — a formal account of how
  optimizing against reward learned from biased human preferences increases the
  tendency to affirm the user's stated belief.

## 3. Token cost — replay is quadratic

This is arithmetic; the primary "evidence" is the formula `Σₙ₌₁..T n·s =
s·T·(T+1)/2` for full-history replay versus a capped-memory model. There is no
paper needed for the math; the honest nuance (unbounded-but-compressed memory is
still O(T²) with a tiny constant; *bounded* memory is truly O(T)) is discussed in
`docs/architecture.md` and modeled in `benchmarks/token_cost/`.

## Reading this table honestly

* The context-rot result is replicated, frontier-model-wide, and this framework's
  core bet is directly supported by it — Chroma's own framing ("the models are
  fully capable; the failure is that the working context doesn't stay clean") is
  close to a summary of this repo's thesis.
* The sycophancy/self-anchoring result is *directional support*, not a measured
  effect of this framework. The benchmark ships to measure it.
