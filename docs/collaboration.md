# Compute, collaborators, authorship — what to ask for, from whom, in what order

Short version: **compute is not the bottleneck; a clean pilot result is.** Run
the pilot yourself for the price of a lunch, apply to the two free credit
programs, and only then approach specific researchers — with numbers, a named
confound you cannot rule out alone, and an authorship plan written down.

## 1. Do the arithmetic before asking anyone for anything

`python -m benchmarks.budget` builds the exact prompts the benchmarks send and
prices them. Default campaign (context-rot at 200 / 1 000 / 5 000 turns,
FlipFlop with 8 items × 4 rounds, **5 seeds**, 5 models):

| Model | Raw (full transcript) | Memory condition | Total |
|---|---|---|---|
| openai/gpt-4o | $16.97 | $0.47 | $17.44 |
| anthropic/claude-3.5-sonnet | $20.41 | $0.61 | $21.02 |
| qwen/qwen-2.5-72b-instruct | $2.36 | $0.05 | $2.41 |
| openai/gpt-4o-mini | $1.02 | $0.03 | $1.05 |
| google/gemini-2.0-flash-001 | $0.68 | $0.02 | $0.70 |
| **all five** | | | **≈ $43** |

Notice where the money goes: >95 % of the bill is the *raw* baseline — the
thing the framework exists to remove. The memory condition is ~500 tokens per
call regardless of history length.

Scaling up to a benchmark reviewers recognise — LongMemEval-S is 500 questions
over ~115 k tokens of history each, where long-context models drop 30–60 %
versus oracle retrieval — the raw baseline alone is ~57 M input tokens per
model: roughly **$150–200 per frontier model, ~$6 on Gemini Flash**, plus one
distillation pass (use a cheap model: <$10). A three-frontier-model campaign is
**$500–700 for one seed.** Still not university-cluster money; still inside two
credit grants.

So "we need compute" is the wrong opening line. It signals the arithmetic
wasn't done. What you actually lack is listed in §3.

## 2. Free credits — apply now, in parallel, no collaborator needed

| Program | What | Cadence | Fit |
|---|---|---|---|
| Anthropic External Researcher Access | $1 000 API credits, standard models, **safety/alignment topics only** | reviewed first Monday of each month | Sycophancy / self-anchoring under replay is squarely an alignment topic. Lead with H2, not token cost. |
| OpenAI Researcher Access Program | up to $1 000 API credits, valid 12 months | reviewed quarterly (Mar / Jun / Sep / Dec) | Frame as "responsible deployment: does replaying the model's own outputs amplify sycophancy?" |
| OpenRouter, your own card | pay-as-you-go | now | $20–50 runs the whole pilot in §4 today. Do not wait for the grants for this. |
| IndiaAI Mission compute | subsidised GPUs (~₹65/GPU-hr) for registered startups / academic researchers | ongoing | **Only** relevant for the open-weights leg (§3, item 4). Irrelevant for API benchmarks. |

Application blurb (≈100 words, adapt):

> We test whether replaying an LLM's own prior outputs in context — the default
> in every chat and agent loop — amplifies sycophantic capitulation and
> self-anchoring, and whether a compact, structured memory that withholds prior
> assistant text (but preserves conclusions) removes the effect. We use a
> SYCON-style FlipFlop protocol (Turn-of-Flip, Number-of-Flip) and Chroma-style
> focused-vs-full context tests across ≥3 model families, ≥5 seeds, with paired
> statistics. Harness, baselines and offline mock are open-source (MIT):
> github.com/XLORD-oss/Agent-Memory. Estimated usage: ≈$X (see
> `benchmarks/budget.py` output attached).

## 3. What a collaborator is actually for

Rank-ordered by how much it changes the paper's fate:

1. **Experimental-design review.** Someone who has published on sycophancy or
   long-context evaluation will find the confound in an afternoon that would
   otherwise be Reviewer 2's first paragraph. Known ones already: the memory
   condition changes *both* what is shown and *how much* is shown — you need a
   length-matched control (same token count, random transcript slice) to
   separate "less clutter" from "no self-replay".
2. **Baselines reviewers demand.** RAG over the transcript (top-k chunks),
   rolling LLM summary, and a MemGPT/Letta-style agent. Without them the paper
   is "our thing vs. the strawman". A grad student can build these in a week
   from `benchmarks/common/harness.py`.
3. **Statistics and judging.** Seeds, paired bootstrap CIs, multiple-comparison
   correction; human labels on a subset to validate the exact-match / flip
   detector. Boring, decisive.
4. **Open-weights runs for the calibration story.** The self-anchoring
   hypothesis is really about *confidence* drift, and confidence needs
   token log-probs. Closed APIs give partial or no logprobs; open models
   (Qwen-72B, Llama-70B) on a lab cluster give full logprobs, hidden states,
   and thousands of cheap seeds. **This is the one place university compute
   genuinely matters** — and where IndiaAI credits or a lab's GPUs are the
   right ask.
5. **arXiv endorsement and venue know-how.** Your endorsement is in
   physics/nlin categories; cs.CL / cs.AI need a separate endorser. An NLP
   co-author solves this in one click and knows which venue's reviewers care
   about which baseline.

Approach **people, not universities.** Highest-yield targets are authors of the
papers the repo already builds on — SYCON-Bench, the self-anchoring
calibration-drift paper, LongMemEval, Chroma's context-rot report — plus one
local NLP group (Hyderabad has IIIT-H's language-technology centre in
commuting distance; a local student baseline-builder is worth more than a
remote famous name). Your own department is also an asset: dynamics people
understand state estimation, and a "memory as data assimilation" framing gets
you a co-supervisor and a reader, even if the venue is NLP.

## 4. Sequence — do not skip step 1

| Step | Cost | Output |
|---|---|---|
| 0. Fill the two TODOs in `CITATION.cff`, get an ORCID (2 min, free) | $0 | Citable software |
| 1. **Pilot:** Gemini Flash + GPT-4o-mini + one frontier model, 5 seeds, both benchmarks | $20–50 | The only artefact anyone senior will read |
| 2. Apply to both credit programs with the pilot numbers attached | $0 | ~$2 000 of runway |
| 3. Write a 2-page pilot note: hypothesis, protocol, numbers with CIs, the confound you can't close alone | 1 day | The thing you attach to a cold email |
| 4. Email 3–5 specific researchers (template below) | $0 | Ideally one design-review call |
| 5. Agree authorship in writing (CRediT roles) **before** anyone spends time | 1 email | No dispute at submission |
| 6. Baselines + stats + open-weights calibration leg | their students / credits | The paper |

If step 1 shows no gap on real models, you have saved everyone's time and you
still have a useful engineering tool and a negative result worth a blog post.
Do not approach anyone before you know which of those two worlds you are in.

Cold email (≈120 words; one paper of theirs, one number of yours, one ask):

> Subject: Does replaying the model's own outputs drive the flips you measured in SYCON-Bench?
>
> Dr ___, your ToF/NoF metrics are the backbone of a small pilot I ran:
> full-history vs. a compact memory that withholds prior assistant text but
> keeps the stated conclusion. On [model] over 5 seeds, flip rate went
> [a → b] (paired bootstrap 95 % CI [ ]). Harness and data are open
> (github.com/XLORD-oss/Agent-Memory).
>
> The confound I can't rule out alone is length vs. content — I'd value 30
> minutes on a length-matched control design, and if it holds up I'd propose
> a joint paper with authorship agreed up front (I'd expect to do the
> engineering and first draft). My background is nonlinear dynamics /
> weather predictability; this started as a state-estimation problem.

## 5. Co-authorship vs. software citation — the decision rule

They are different transactions; don't trade one for the other.

**Authorship follows intellectual contribution to *that paper*.** In both
directions:

* Someone who redesigns the protocol, builds baselines, runs the stats, or
  writes sections → **co-author.** Someone who supplies $1 000 of credits and
  nothing else → **Acknowledgments.** Access is not authorship; the ICMJE /
  CRediT norms are explicit, and "authorship for compute" is a known smell.
* Conversely, a lab that *uses* Agent-Memory in their own study owes you a
  **software citation** (that is what `CITATION.cff` is for), not a byline.
  Asking for co-authorship as a condition of tool use is the fastest way to
  get them to pick a different memory library. If you actively help design or
  run *their* experiment, then you have contributed and authorship is on the
  table — earned, not levied.

**Order and roles, agreed before work starts, in writing:**

* You did the framework, the pilot, and will write the first draft → **first
  author**. A senior collaborator who supervises the study and secures the
  rigor → **last author.** Students who build baselines → **middle authors.**
* Use the CRediT taxonomy (Conceptualization, Methodology, Software, Formal
  analysis, Writing – original draft, Supervision…) — a table in the email
  thread is enough; it prevents 90 % of disputes.
* Software citation is what you ask from *everyone else*: users, blog posts,
  derived works. Make it frictionless (`CITATION.cff` → GitHub's "Cite this
  repository" button) and it will happen; ask for it as a favour and it won't.

## 6. What to say and not say

Say: "hypothesis", "pilot", "confound", "paired CI", "we withhold prior
assistant text and keep conclusions", "length-matched control".

Don't say: "reduces sycophancy" (not shown), "publishable" (not yet),
"skyrockets intelligence" (never), "needs compute" (it needs $43).

The people worth collaborating with are exactly the ones who will check.
