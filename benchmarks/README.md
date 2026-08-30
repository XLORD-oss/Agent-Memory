# Benchmarks

The framework makes two falsifiable behavioral claims and one arithmetic claim.
Each gets its own experiment, built on a **published methodology** rather than a
home-grown eval, so the results are defensible in front of the audience that
matters:

| Benchmark | Claim under test | Methodology | Offline? |
|---|---|---|---|
| [`context_rot/`](context_rot/README.md) | Compact memory recovers accuracy lost to long-context degradation (the 30–50% effect Chroma measured across 18 frontier models) | Chroma's focused-vs-full-prompt design (research.trychroma.com/context-rot) | ✅ `--mock` |
| [`sycophancy/`](sycophancy/README.md) | Keeping prior assistant outputs out of context reduces self-anchoring / sycophantic flipping under sustained push-back | FlipFlop / SYCON Bench ToF & NoF metrics (arXiv:2505.23840) | ✅ `--mock` |
| [`fidelity/`](fidelity/README.md) | The compact memory is a lossy compression — measure its *assimilation error*: state fidelity vs cap (rate-distortion), distillation loss, usage-protected retention | Reduced-order-model fidelity; no model called, scores the memory files directly | ✅ always |
| [`token_cost/`](token_cost/README.md) | Token cost is quadratic under replay, near-linear under compact memory | Arithmetic (no model needed) | ✅ always |

Every harness runs two conditions over the **same task, same model, same script**:

* **full / raw** — the entire raw transcript is replayed into context (the naive default).
* **memory** — the transcript is distilled into the compact memory file and only that
  is placed in context.

The mock model (`--mock`) deterministically emulates the documented failure modes
(lost-in-the-middle, length rot, sycophantic flipping) so the harness is fully
testable in CI without an API key. Swap in a real model with `--model` and the
exact same code path runs against the real thing — the experiment is designed so
the number you get back is the number you can stand behind.

## Quick start

```bash
pip install -e .            # from repo root
python -m benchmarks.token_cost.model
python -m benchmarks.context_rot.run --mock
python -m benchmarks.sycophancy.run_flipflop --mock
```

## Real model runs

```bash
export OPENAI_API_KEY=...
python -m benchmarks.context_rot.run --model gpt-4o-mini --turns 200 --facts 10
python -m benchmarks.sycophancy.run_flipflop --model gpt-4o-mini --questions 8 --rounds 4
```

Results land in `runs/*.json` (git-ignored). See each subpackage's README for
the full methodology and how to read the numbers.
