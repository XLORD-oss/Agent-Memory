# Context-rot benchmark

**Question:** for the same task and same model, does replacing the full raw
transcript with the compact memory file recover accuracy lost to long-context
degradation?

**Why this methodology:** Chroma's 2025 report (*Context Rot*, Hong, Troynikov &
Huber — [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot))
showed all 18 frontier models degrade as input grows — 30–50% accuracy drops from
context clutter well before the window is full, with a large gap between *focused*
prompts and *full* prompts on conversational tasks. Rather than inventing our own
eval, we adapt that exact focused-vs-full design to this framework's two conditions.

## Setup

1. A synthetic long conversation (`--turns`, default 200) full of topic drift and
   distractors, with `--facts` ground-truth needles planted at scattered positions.
2. **Condition `raw`:** each question is asked with the entire transcript in context.
3. **Condition `memory`:** the transcript is distilled into `memory.md` (facts only,
   via the benchmark's deterministic `MarkerDistiller`) and the same questions are
   asked with memory + a small rolling window.
4. Exact-match accuracy is scored per condition; the gap is reported.

## Run

```bash
python -m benchmarks.context_rot.run --mock                       # offline demo
python -m benchmarks.context_rot.run --model gpt-4o-mini --turns 200 --facts 10
```

## Reading the output

* **Gap (memory − raw)** — the framework's context-rot number. A large positive gap
  means memory recovers accuracy that the transcript wastes.
* **Input tokens** — memory should send a small fraction of the raw bill.

## Caveats baked into the design

* The mock emulates the documented failure modes so CI can exercise the harness;
  treat mock numbers as a methodology demonstration, not a result.
* Distillation in the benchmark is deterministic (facts extracted verbatim) to
  isolate the *presentation* effect. Production runs can swap in `LLMDistiller`.
* Exact-match scoring is strict; a real deployment should add an LLM judge for
  paraphrase-tolerant scoring (see docs/roadmap.md).
