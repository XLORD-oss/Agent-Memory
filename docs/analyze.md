# analyze_trace — measure against your real usage

`TraceAnalyzer` replays a real chat log through the engine under any policy and
quantifies the two things that matter: **how many tokens you actually send** and
**which facts earn their keep**. It runs fully offline — no API credits.

## CLI

```bash
# A single profile, with the per-turn token table, usage report, evictions
python -m agent_memory.analysis --trace logs/session.jsonl --profile general

# All four profiles side by side
python -m agent_memory.analysis --trace logs/session.jsonl --compare

# Chart the per-turn curve (needs matplotlib)
python -m agent_memory.analysis --trace logs/session.jsonl --profile coding --plot cost.png

# Built-in demo session, repeated to a realistic length
python -m agent_memory.analysis --trace demo --repeat 5 --compare

# Write the machine-readable report
python -m agent_memory.analysis --trace logs/session.jsonl --out report.json
```

## Trace formats

| Format | Example |
|---|---|
| JSONL (OpenAI-style) | `{"role": "user", "content": "..."}` / `{"role": "assistant", "content": "..."}` |
| JSONL (paired) | `{"user": "...", "assistant": "..."}` |
| Plain text | `user: ...` / `assistant: ...` lines |

Import from OpenAI / Claude / any export → paste into a `.jsonl` → run.

## What it reports

1. **Totals.** Naive full-history replay tokens vs compact-memory tokens, ratio,
   % saved. (This is the O(T²)-vs-O(T) number, measured on *your* session.)
2. **Last-turn context.** The two numbers that matter for latency: naive last
   turn vs compact last turn. Compact should stay roughly flat.
3. **Memory footprint.** Entries created / archived / surviving, and the
   compression ratio vs the raw transcript.
4. **Usage report.** Which entries the model's answers actually referenced
   (`uses`). Zero-referenced entries are your eviction candidates — the
   usage-weighted policy archives them first.
5. **Eviction list.** What got archived and in which order.
6. **Per-turn series** (and optional plot): watch compact stay flat while naive
   climbs.

## Interpreting the profile comparison

Profiles trade working-memory precision for token savings. The `--compare`
table makes the tradeoff explicit:

```
| Profile | Compact total | Naive total | Ratio | Saved |
| general | 21,687  | 77,080 | 0.281 | 71.9% |
| coding  | 45,382  | 77,080 | 0.589 | 41.1% |
```

`coding` spends more tokens to keep recent code in context — its max context/turn
is larger. On a tasks-that-need-it basis that is the right spend; on a long
chatty session it's waste. Your trace tells you which regime you live in.

## Programmatic use

```python
from agent_memory.analysis import TraceAnalyzer, compare_policies, load_trace

turns = load_trace("logs/session.jsonl")
result = TraceAnalyzer(profile="research", memory_cap_tokens=4000).analyze(turns)
print(f"saved {result.saved_pct:.1f}% of tokens")
print(result.usage)

matrix = compare_policies(turns)   # {profile: {ratio, saved_pct, ...}}
```

## Honest caveats

* **Distillation yield is real.** Without the `LLMDistiller`, the runner uses
  `RuleDistiller` (explicit "remember/prefer/decided" patterns). Real logs with
  implicit facts will under-report `entries_created`. For production numbers,
  pass an `LLMDistiller` (needs an API key) or pre-distilled turns.
* **Short traces inflate working-memory cost.** On a <20-turn session the
  bounded working window can duplicate distilled content (compact > naive for
  `coding`/`writing`). The crossover is where the framework starts winning —
  which is itself a useful number to know about *your* session length.
* **Usage tracking is a heuristic** (token overlap at `usage_threshold=0.4`).
  Good enough to rank entries; it is not a retrieval-quality measure.