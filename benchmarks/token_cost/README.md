# Token-cost model

**Claim:** full-history replay processes O(T²) tokens over a T-turn conversation;
compact memory processes ~O(T).

This is arithmetic, not a model measurement, so it needs no API key and runs in CI.

* Full-history replay: turn N re-sends N−1 prior turns → `per_turn · T·(T+1)/2`.
* Compact memory: the active context is bounded (a capped fact store + a small
  rolling window of recent raw turns) → roughly linear in T.

```bash
python -m benchmarks.token_cost.model                     # table
python -m benchmarks.token_cost.model --plot cost.png     # requires matplotlib
```

## Representative table (default assumptions: ~500 tokens/turn, 3000-token memory cap)

| Turns | Full history (tokens) | Compact memory (tokens) | Ratio | Saved |
|---|---|---|---|---|
| 50 | 637,500 | 179,488 | 0.282 | 71.8% |
| 200 | 10,050,000 | 929,488 | 0.092 | 90.8% |
| 1,000 | 250,250,000 | 4,929,488 | 0.020 | 98.0% |
| 5,000 | 6,251,250,000 | 24,929,488 | 0.004 | 99.6% |

The saving grows with conversation length because memory growth is capped by
archiving while replay growth is not capped at all.
