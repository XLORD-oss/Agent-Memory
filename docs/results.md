# Results

> **Placeholder.** Run the benchmarks against real frontier models to fill this in.
> See `docs/running.md` for one-line commands.

## Context-rot benchmark

| Model | Raw accuracy | Memory accuracy | Gap | Token savings | 
|---|---|---|---|---|
| mock (offline) | 30.0% | 100.0% | +70.0 pts | 8.1% of raw |
| gpt-4o-mini | — | — | — | — |
| gpt-4o | — | — | — | — |
| claude-3-haiku | — | — | — | — |
| claude-3.5-sonnet | — | — | — | — |
| gemini-2.0-flash | — | — | — | — |
| | | | | |

## Sycophancy (FlipFlop) benchmark

| Model | Full-history flip rate | Memory flip rate | Full mean ToF | Memory mean ToF |
|---|---|---|---|---|
| mock (offline) | 100% | 0% | 1.75 | ∞ (never) |
| gpt-4o-mini | — | — | — | — |
| gpt-4o | — | — | — | — |
| claude-3-haiku | — | — | — | — |
| claude-3.5-sonnet | — | — | — | — |
| gemini-2.0-flash | — | — | — | — |

## Token-cost model

| Turns | Full history tokens | Compact memory tokens | Ratio | Saved |
|---|---|---|---|---|
| 50 | 637,500 | 179,488 | 0.282 | 71.8% |
| 100 | 2,525,000 | 429,488 | 0.170 | 83.0% |
| 200 | 10,050,000 | 929,488 | 0.092 | 90.8% |
| 500 | 62,625,000 | 2,429,488 | 0.039 | 96.1% |
| 1,000 | 250,250,000 | 4,929,488 | 0.020 | 98.0% |
| 2,000 | 1,000,500,000 | 9,929,488 | 0.010 | 99.0% |
| 5,000 | 6,251,250,000 | 24,929,488 | 0.004 | 99.6% |

## Methodology

See `benchmarks/context_rot/README.md` and `benchmarks/sycophancy/README.md` for the full methodology. Every benchmark runs the same task, same model, two conditions — the only variable is whether context is the full raw transcript or the compact memory file.