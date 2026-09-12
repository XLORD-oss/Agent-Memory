# Training under the memory contract — is the contract a training variable?

Status: **experiment design + data tooling. No model has been trained.**
Everything below is a hypothesis with a protocol, not a result.

## The three things "train the model with this framework" could mean

| | What is learned | Prior work | Verdict |
|---|---|---|---|
| **A. Train the consumer** | Condition the model on the compact memory (not the transcript) during fine-tuning, identical targets | none isolates this | **the experiment** — see below |
| B. Train the producer | RL the distiller / memory manager (ADD/UPDATE/DELETE, what to keep) | Memory-R1 (2508.19828), Mem-α (2509.25911), ECHO | crowded; our fidelity benchmark is a free reward signal — later plug-in, not the lead |
| C. Evolve the context | Treat the memory as a self-improving playbook, no weight updates | ACE (ICLR 2026, 2510.04618), Dynamic Cheatsheet | our map already does this at inference; not a differentiator |

A is the one worth a paper because it turns the inference-time claim (H2:
withholding prior assistant text reduces capitulation) into a question about
**what the weights learn**. Every chat model today is trained on
`(full transcript → next reply)`. Nobody has asked what changes when the same
targets are learned under `(compact memory → next reply)`.

## Hypothesis

> **H4.** A model fine-tuned under the memory contract — never conditioned on
> its own prior replies, always conditioned on a compact distilled state — will
> (a) capitulate less under multi-turn push-back and (b) show smaller
> confidence drift when later shown its own outputs, *even when evaluated with
> a full-transcript prompt*.

Falsifiable in both directions, and both answers are useful:

* **H4 holds** → the contract is a training variable; the effect is in the
  weights, and the inference-time framework becomes a *data-generation* recipe.
* **H4 fails** (memory-trained model behaves like the others once handed a
  transcript) → it is a prompt-format effect. That is still the H2 result, and
  it tells labs the fix is cheap: change the prompt, not the training.

## The 2×2 design (`agent_memory.export`)

The data tool rebuilds any chat trace into four training sets with
**identical completions** and different prompts:

| | long context | short context |
|---|---|---|
| **replays own outputs** | `full` — today's default | `matched` — recency-truncated to the memory prompt's token count |
| **withholds own outputs** | `user_only` — assistant turns dropped, everything else kept | `memory` — the compact distilled state |

The two off-diagonal cells are the controls that make the result mean
something. Without `matched`, "memory" is confounded with "shorter". Without
`user_only`, it is confounded with "distilled". With both, the main effects of
*length* and *self-replay* separate, and their interaction is measurable.

```bash
python -m agent_memory.export traces/*.jsonl --out data/sft            # all four
python -m agent_memory.export --demo --out /tmp/sft                      # offline preview
python -m agent_memory.export traces/*.jsonl --out data/sft --system contract   # anti-sycophancy system prompt in all cells
```

Demo trace, 12 turns (from `summarize`):

| Condition | Prompt tok (mean) | Seq tok (total) | vs full | Prompts replaying own output |
|---|---|---|---|---|
| full | 285 | 3,644 | 100 % | 92 % |
| user_only | 161 | 2,156 | 59 % | 0 % |
| memory | 168 | 2,237 | 61 % | 0 % |
| matched | 147 | 1,991 | 55 % | 92 % |

Rules the exporter enforces, because each one is a known confound:

1. **Loss on `completion` only.** Training on whole sequences would teach the
   `full` model to reproduce its *earlier* replies too — that is a second
   treatment, not the one under test. TRL's prompt–completion format does this
   by default; Axolotl needs `train_on_inputs: false`.
2. **Same system prompt in every cell.** Default is a neutral one. The memory
   contract prompt contains anti-sycophancy instructions — a real variable, so
   toggle it deliberately (`--system contract`) across *all* cells, never in
   one cell only.
3. **Same targets, same turns, same order.** One target per (trace, turn);
   nothing is dropped from one cell that is kept in another.
4. **`matched` is well-formed.** Truncation keeps the system prompt and the
   current user message and never opens with a dangling assistant reply.

## Protocol

**Base.** One open instruct model with logprobs: Qwen2.5-7B-Instruct or
Llama-3.1-8B-Instruct. LoRA (r=16, all linear), 1–2 epochs, identical
hyperparameters in all four cells, 3 seeds each → 12 adapters.

**Data.** Real multi-turn traces, ≥ 5 turns, the same set for every cell.
Sources: your own exported sessions (`analyze_trace` already loads them),
LMSYS-Chat-1M / WildChat multi-turn subsets, or a synthetic set built with the
harness. Target 5–20 k conversations; it is the *pairing*, not the volume, that
carries the design. Completions that are themselves sycophantic capitulations
will teach capitulation in every cell equally — filter them or accept the
dilution.

**Evaluation — the cross-eval matrix.** Every adapter × every *test-time*
prompt format:

|  train \ test | full | user_only | memory | matched |
|---|---|---|---|---|
| full | · | · | · | · |
| user_only | · | · | · | · |
| **memory** | **the cell that decides H4** | · | · | · |
| matched | · | · | · | · |

Metrics per cell, `benchmarks/sycophancy` protocol (SYCON-style):

* **Turn-of-Flip / Number-of-Flip** under 4 rounds of push-back.
* **Calibration drift**: token-logprob confidence on the answer at round 0 vs.
  after the model has been shown its own (correct) prior reply — the
  self-anchoring measurement that closed APIs cannot give you.
* **Task retention**: held-out accuracy on the same traces' factual questions,
  so a "less sycophantic" adapter that simply got worse is caught.
* **Length control check**: `matched` vs `full` isolates length; `user_only`
  vs `full` isolates self-replay; `memory` vs `user_only` isolates distillation.

Report paired bootstrap CIs across seeds; the story is the *difference between
rows*, not any single number.

## Cost — and why this is the first thing that needs a GPU

| Item | Rough cost |
|---|---|
| Data export (this module) | $0, CPU, seconds per thousand conversations |
| LLM distillation pass for `memory` cell, 10 k conversations × ~150 tok | ≈ $5–15 on a cheap model |
| 12 LoRA runs, 7–8B, ~10 M tokens each (the `full` cell is the expensive one) | ~40–80 A100/H100-hours total |
| Cross-eval: 12 adapters × 4 formats × sycophancy suite with logprobs | ~10 GPU-hours |
| **Total** | **≈ 50–100 GPU-hours** — a weekend on 2 GPUs, or ~₹3 000–6 500 at IndiaAI rates |

This is the leg where a university collaborator (see `collaboration.md`) is not
a courtesy but a necessity: local weights for logprobs, a cluster for 12
adapters, and someone who has done LoRA sweeps before so the hyperparameters
are not the finding.

## What it does *not* show

* Nothing about pre-training or RLHF: the contract is applied at SFT time on
  an already-aligned model. If RLHF is where sycophancy is amplified
  (2602.01002), SFT can at best partially counteract it.
* Nothing about long-horizon agents: traces are conversations, not tool loops.
* Nothing without the controls: a `memory`-vs-`full` two-cell run would be
  uninterpretable and should not be run.

## Reading order

`export.py` (the data) → this file (the design) → `benchmarks/sycophancy/`
(the metrics) → `collaboration.md` (who runs the GPUs and how authorship is
agreed before they do).
