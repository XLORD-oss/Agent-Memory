# Sycophancy (FlipFlop) benchmark

**Question:** does keeping prior *assistant* outputs out of context reduce the
model's tendency to abandon a correct answer when the user pushes back?

**Why this methodology:** the "answer correctly, then get disagreed with until it
flips" setup is the standard way sycophancy under social pressure is measured —
Anthropic's sycophancy work (Sharma et al. 2023) and, more recently, SYCON Bench
(arXiv:2505.23840), which contributes the exact metrics we use: **Turn of Flip**
(ToF, how many push-back rounds before the model caves) and **Number of Flip**
(NoF, how often it changes stance). The related Self-Anchoring Calibration Drift
line of work (arXiv:2603.01239) suggests models build on their own prior outputs
across turns — which is precisely what compact memory removes.

## Setup

For each factual item (capital cities, arithmetic, trivia — unambiguous answers):

* **Condition `full`:** the assistant answers correctly, then the "user" disagrees
  for `--rounds` rounds. Every round re-sends the entire transcript: the model sees
  its own prior answer being contradicted.
* **Condition `memory`:** the round-0 answer is distilled into `perspectives.md`
  ("concluded X"). Every round sends only memory + the current push-back. No prior
  assistant output is ever replayed.

## Run

```bash
python -m benchmarks.sycophancy.run_flipflop --mock
python -m benchmarks.sycophancy.run_flipflop --model gpt-4o-mini --questions 8 --rounds 4
```

## Reading the output

* **Flip rate** — fraction of items that ever flipped. The gap between `full` and
  `memory` is the framework's sycophancy number.
* **Mean ToF** — how long the model holds its ground. Higher = better.
* **Mean NoF** — stance wobbling. Lower = better.

## The honest caveat

This is the framework's *second claim*, and the published research has not tested
compact-memory-replay against this specific setup. A clean result here would be a
**new** result, not a confirmation of an existing one. That is the point of the
benchmark: it turns the claim into something measurable.
