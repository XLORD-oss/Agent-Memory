# What exactly we ask a university for — statement of work

This is the one-page ask, then the detail behind each line. It assumes the
inference pilot (`kaggle.md`) has been run and its numbers are attached; the
ask is much weaker without them. Authorship is agreed up front: the PI's group
brings design review, compute, baselines and the writing standard; that is a
first/last-author collaboration, not a favour.

## The one-page ask

> **Project.** Does replaying an LLM's own prior outputs in context drive
> sycophantic capitulation and self-anchoring — and does a compact memory
> that withholds them (while keeping the stated conclusion) remove it, at
> inference time and/or when baked into training?
>
> **What exists.** Open-source framework + benchmark harness (MIT), 117
> offline tests, pilot results on 7–27B open models with logprobs (attached),
> a 4-arm controlled sycophancy design and a 2×2 SFT data exporter.
>
> **What we ask for (12 weeks):**
>
> | # | Item | Cost to you |
> |---|---|---|
> | 1 | One experimental-design review meeting (90 min) before any compute is spent | 1 senior researcher, 90 min |
> | 2 | GPU allocation: **~350 A100/H100-hours** (breakdown below), burstable, over 8 weeks | cluster time |
> | 3 | One graduate student at ~40 % for 12 weeks who **owns** the central study: baselines, item set + labels, the H4 fine-tuning grid, the analysis, the first draft | ~0.4 FTE × 12 wk |
> | 4 | Access for one external collaborator (me) to the cluster and the group's eval stack | an account |
> | 5 | arXiv cs.CL endorsement and venue targeting (COLM / EMNLP / TMLR) | 10 min |
>
> **What you get.** A co-authored paper with a falsifiable claim in both
> directions, a released benchmark others will run, and a fine-tuning
> result (H4) that no one has reported: whether the memory contract is a
> *training variable*.
>
> **Roles.** The framework, harness and pilot are done and provided as-is.
> Your student leads the central study (H4 fine-tuning and cross-eval), the
> baselines, and the analysis, and drafts the paper; I integrate, maintain
> the harness, run the inference campaign with them, and co-write.
> Authorship: student first, me second, [PI] last — settled today in the
> CRediT table (§6), revisited once against actual contribution when the
> main results are in (`authorship.md`).

## 1. The review meeting — what we ask them to attack

The single highest-value hour. Send `claim.md`, the pilot tables and this file
72 h before. Ask them specifically to break:

* **The confound structure.** The 4-arm design (`full` / `memory` /
  `user_only` / `truncated`) separates self-replay from length from
  distillation. Is a fifth arm needed (e.g. *paraphrased* self-replay, to
  test whether it is the model's *exact words* or its *stance* that anchors)?
* **The flip detector.** Regex-based `is_flip` on one-value answers. They
  will want a human-labelled subset and a judge-model agreement number; the
  student does this (item 3).
* **Item set.** 8 trivia items is a pilot, not a study. Ask which existing
  set to adopt so the number is comparable: SYCON-Bench's debate/ethics
  items, TruthfulQA-derived factual items, or a math set (GSM8K-style) where
  the correct answer is unambiguous and push-back is unarguably wrong.
* **Which confidence measure.** We report exp(mean token logprob) of the
  reply. They may prefer P(correct answer token | prompt) at the first
  answer position, or verbalised confidence. Decide once, before running.
* **Model choice for H4.** Qwen2.5-7B vs Llama-3.1-8B vs an instruct model
  *before* RLHF (e.g. an SFT-only checkpoint), since the hypothesis is
  partly about what RLHF amplified.

Walk out with a frozen pre-registration: hypotheses, arms, item set, metrics,
seeds, analysis plan, stopping rule. Post it (OSF or a dated commit) before
the big runs. This costs nothing and is worth a full tier at review.

## 2. The compute — exactly what runs and what it costs

All figures are estimates for one 7–8B model family; multiply by the number of
families. Times are A100-80GB-hours; H100 ≈ 0.6×.

### 2a. Inference at scale (the pilot, properly powered)

| Run | Calls | Notes | GPU-h |
|---|---|---|---|
| Sycophancy, 4 arms, **200 items** × 6 rounds × **10 seeds** | 4 × 200 × 7 × 10 ≈ 56 k | short prompts, vLLM batched | ~6 |
| Same with `--system neutral` (prompt ablation) | 56 k | | ~6 |
| Context-rot at 200 / 1 k / 5 k / 20 k turns, 10 seeds | 4 × 20 × 10 = 800 | raw arm at 20 k turns ≈ 430 k tokens/call — needs a 1M-context open model or cap at 5 k | ~10 |
| LongMemEval-S, 500 questions, raw vs memory vs baselines (RAG, rolling summary) | 2 k | 115 k tokens/call on raw arm | ~25 |
| LLM distillation pass for the memory arm (`LLMDistiller`) | per turn | small model | ~5 |
| **Subtotal per family** | | | **~50** |
| × 3 families (Qwen2.5-7B, Llama-3.1-8B, Qwen3.8-27B) | | | **~150** |

### 2b. H4 — fine-tuning under the contract

| Run | Notes | GPU-h |
|---|---|---|
| SFT data export (CPU) | `agent_memory.export` on ~10 k multi-turn conversations | 0 |
| 4 cells × 3 seeds = 12 LoRA adapters, 7–8B fp16, r=16, ~10 M tokens each | `full` is the long-context cell and dominates | ~80–120 |
| Cross-eval: 12 adapters × 4 test formats × sycophancy suite (with logprobs) + task-retention eval | | ~20 |
| Held-out check: 2 adapters on a second base model (does it transfer?) | | ~30 |
| **Subtotal** | | **~130–170** |

### 2c. Slack

Reruns after the review meeting changes something, a failed sweep, one more
seed because a CI straddles zero: **~30–50**.

**Total: ~310–370 A100-hours.** Round to **350** in the ask. On a university
cluster this is a small allocation — a few days of one 8-GPU node — which is
the point: we are asking for a *specific, bounded* amount tied to a
pre-registered plan, not "access".

If they cannot give it all: **2a alone (150 h) produces a publishable
inference-only paper; 2b is what turns it into the paper worth a main
conference.** Say so; it lets them size the commitment.

## 3. The student — a concrete 12-week plan they own

This is a first-author plan: the student makes the design calls in each row
(with review), and the code lands in their own modules
(`benchmarks/baselines/`, `training/`, `analysis/`) with their name on it.
See `authorship.md` for why the split is structured this way.

| Weeks | Task (student decides; I review) | Output |
|---|---|---|
| 1 | Draft the pre-registration from the review meeting: hypotheses, arms, item set, metrics, seeds, analysis plan, stopping rule | dated pre-registration |
| 2–3 | Baselines as registered arms in `benchmarks/baselines/`: RAG-over-transcript (top-k, same token budget as memory), rolling LLM summary (stub provided), MemGPT/Letta-style; decide what "fair" means for each | 3 arms selectable with `--arms` |
| 3–4 | Measurement: adopt the item set chosen at review; 300 human flip labels; judge-model κ; choose the confidence metric | `data/items.jsonl`, agreement table |
| 5–6 | Inference campaign (2a) with me; own the stats: paired bootstrap, sign-flip tests, multiple-comparison correction, power check against the pilot effect | `results/` tables, analysis notebook |
| 7–10 | H4: choose base model + LoRA config, export the 2×2 data, run the 12 adapters, cross-eval matrix, task-retention check; own every failed run | `training/`, 12 adapters, the matrix |
| 11–12 | First draft: Methods, Results, the figure that carries the paper; I co-write Discussion and Related Work | draft |

The student learns the sycophancy/eval literature end to end, owns the
paper's central experiment, and has the natural follow-up (paraphrased-replay
arm, RLHF-vs-SFT) already in view — mention that to the PI; it is how a good
PI decides.

## 4. What we do *not* ask for

* Money, salary, or travel. (If they have conference travel funds for the
  student, great; not part of the ask.)
* Their proprietary data or models.
* Exclusive rights — code stays MIT, the benchmark is released regardless.
* Open-ended "collaboration". Everything above has a week number.

## 5. Who to send this to, and in what order

1. **Sycophancy / evaluation groups first** — authors of SYCON-Bench, the
   multi-turn sycophancy measurement papers, the self-anchoring calibration
   work. They own the metric; they can veto the design in one email, which
   is exactly the review we need.
2. **A local group second** (IIIT-H language technologies, IIT-H, IISc) for
   the student and the cluster — remote review + local compute is a workable
   pairing, and your Hyderabad location makes the local half easy.
3. **Your own department** for the framing: "memory as a reduced-order state
   estimator" is a dynamics story and a co-supervisor who reads it that way
   catches different errors than an NLP reviewer does.

Email: 120 words, one of their papers, one of your numbers with a CI, this
file attached, one ask (the 90-minute review). Do not send the compute
request cold — it comes out of the review meeting, after they have shaped
the plan and therefore own part of it.

## 6. Authorship, written down at the review meeting

CRediT roles, filled in before anyone runs anything (full reasoning in
`authorship.md`):

| Role | Student (1st) | You (2nd) | PI (last) |
|---|---|---|---|
| Conceptualization | ○ | ● | ○ |
| Methodology (study design, pre-registration) | ● | ○ | ● |
| Software — framework, harness, pilot | | ● | |
| Software — baselines, training, analysis | ● | ○ | |
| Investigation — inference campaign | ○ | ● | |
| Investigation — fine-tuning study (H4) | ● | ○ | |
| Data curation (item set, labels) | ● | | |
| Formal analysis | ● | ○ | ○ |
| Writing – original draft | ● | ○ | |
| Writing – review & editing | ○ | ● | ● |
| Supervision | | | ● |
| Resources (compute) | | | ● |

● lead ○ contributing. Order follows from the table. Revisited **once**, when
the main results are in, against what actually happened: if the student led
the central study and analysis, student first; if it turned out otherwise,
co-first (†) or swap — agreed in the same thread as the pre-registration, not
at submission.

## 7. The two-sentence version, for the corridor

"I have a benchmark and a pilot showing that models flip under push-back
mostly when they can see their own previous answer, not just when the context
is long. I need 350 GPU-hours, one student, and your design review to find out
whether that survives scale and whether it can be trained in."
