# The claim, stated precisely

This document exists because the framework's value is easy to overstate and easy
to dismiss. Here is the exact claim, split into what the data already supports,
what is arithmetic, and what is a claim this project is built to *prove*.

## The idea in one paragraph

Long-running agents default to **full-transcript replay**: every turn, the entire
raw conversation is re-sent to the model. That replay is the root of three
independent, compounding problems:

1. **Context rot.** Every frontier model measured degrades as input grows —
   accuracy drops 30–50% from context clutter well before the window is full
   (Chroma, 18 models). The model's *capability* doesn't move; its ability to
   *express* that capability on a cluttered input does.
2. **Quadratic token cost.** Turn N carries ~N turns of tokens, so total tokens
   processed over a T-turn conversation scale as O(T²).
3. **Self-anchoring / sycophantic drift.** The model re-reads its own prior
   answers, which it then treats as settled fact rather than something open to
   revision. Under social pressure it can cave to the user even when wrong.

This framework replaces replay with a **compact, self-maintaining memory**: a
distilled fact store that only grows when a genuinely new fact appears, with
decisions recorded in a perspectives file and older entries split into an
archive. Every turn is assembled fresh from that small store, so the model never
searches a rotting transcript and never re-reads its own raw words.

## What we assert (and what the evidence is)

### 1. Compact memory removes a measured, replicated failure mode — context rot

This is the framework's strongest claim, and it is the one that is already
evidence-backed in the literature:

> Replacing a long, cluttered context with a compact, relevant one recovers
> accuracy that the clutter was destroying — a 30–50% effect, replicated across
> 18 frontier models, on tasks the models solve trivially when the context is
> clean.

Chroma's *Context Rot* report (2025) tested GPT-4.1, Claude 4, Gemini 2.5, Qwen3
and more: every model degraded with input length, on tasks as simple as retrieval
and text replication, and on LongMemEval-style conversations there was a large,
consistent gap between **focused** prompts and **full-history** prompts. The
Lost-in-the-Middle literature (Liu et al., 2023) gives the mechanism: U-shaped
attention, ~15–25 point drops for information in the middle of long contexts.

**The precise wording matters.** We do *not* claim the framework makes a model
smarter. Same weights, same training — the framework changes whether existing
capability gets *expressed* on a given input. "Removes a specific, measured,
30–50% failure mode" is a claim you can put a number on and defend. "Makes the
model smarter" is a claim any skeptical reader can poke at.

### 2. Compact memory cuts long-horizon token cost from quadratic to near-linear

This is arithmetic, not a model measurement:

* Full-history replay: `per_turn · T·(T+1)/2` tokens over T turns → **O(T²)**.
* Compact memory with a capped fact store + small rolling window → **~O(T)**.

At T=1,000 turns with ~500 tokens/turn, replay processes ~250M tokens; compact
memory ~5M (a ~50x reduction). The saving *grows* with conversation length
because memory growth is capped by archiving while replay growth is not.
See `benchmarks/token_cost/` for the full table.

### 3. Keeping prior assistant outputs out of context plausibly reduces self-anchoring and sycophancy — **this is the claim to prove**

The mechanism is real: researchers have documented **Self-Anchoring Calibration
Drift** (SACD) — models' confidence systematically shifts when they build on
their own prior outputs across turns (arXiv:2603.01239) — and sycophancy is a
well-measured failure mode, with RLHF amplifying it and more capable models
showing *more* of it, not less (Sharma et al., 2023; arXiv:2505.23840).

What compact memory does is remove the raw material: the model no longer sees
"I already said X" with all the tone and social momentum that surrounded it. It
sees only the distilled "concluded X". That is a *plausible, real reduction* in
anchoring pressure.

**But nobody has tested compact-memory-replay against this specifically.** The
published studies measure models re-reading their own *raw* turns. So:

* We do **not** assert "drastically reduces bias" as a finding.
* We assert it is **a new, testable hypothesis**, and the repo ships the exact
  experiment to test it (`benchmarks/sycophancy/`): the FlipFlop setup — answer
  correctly, get pushed back on repeatedly, see if it flips — run once with full
  history and once with the memory file. If the gap shows up, that's not a pitch
  anymore; it's a result.

## What we are NOT claiming

* That this makes models more intelligent. (Capability is unchanged; *expression*
  is what moves.)
* That "clean context" is a silver bullet — long-horizon reasoning and tool-use
  still have their own failure modes.
* That compact memory eliminates self-anchoring *drastically* — the evidence for
  the direction is plausible, the magnitude is the open question this repo exists
  to answer.

## The summary

> The framework doesn't claim to upgrade model capability. It claims to remove a
> measured 30–50% expression failure mode (context rot), an O(T²)→O(T) token
> burden, and — pending the benchmark — the self-anchoring drift that comes from
> a model re-reading its own past words. The first is evidence-backed, the second
> is arithmetic, and the third is the experiment this repo ships.
