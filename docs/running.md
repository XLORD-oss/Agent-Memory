# Running the benchmarks

> The sandbox can't reach external APIs, so this is your cheat sheet. Each command
> is designed to be one-liner — the harnesses are built for exactly this.

## One-time setup

```bash
git clone https://github.com/XLORD-oss/Agent-Memory.git
cd Agent-Memory
pip install -e .
# optional: pip install -e ".[llm,plot]"  # for real models + charts
```

## Price the campaign first (no model needed)

```bash
python -m benchmarks.budget                 # default: 3 lengths x 5 seeds x 5 models ≈ $43
python -m benchmarks.budget --turns 200 1000 --seeds 3 --models openai/gpt-4o
```

Builds the exact prompts the benchmarks send, counts tokens, prices per model.
Paste the table into credit applications — see [collaboration.md](collaboration.md).

## Token-cost (no model needed)

```bash
python -m benchmarks.token_cost.model
python -m benchmarks.token_cost.model --plot cost.png  # needs matplotlib
```

## Context-rot benchmark

```bash
# OpenRouter (any model)
OPENROUTER_API_KEY="sk-or-v1-..." \
  python -m benchmarks.context_rot.run \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --turns 200 --facts 10 \
  --out runs/context_rot_gpt4o_mini.json

# Direct OpenAI
OPENAI_API_KEY="sk-..." \
  python -m benchmarks.context_rot.run \
  --model gpt-4o-mini --turns 200 --facts 10

# For multiple models, swap --model:
#   openai/gpt-4o, anthropic/claude-3-haiku, google/gemini-2.0-flash-001
```

## Sycophancy FlipFlop benchmark

```bash
# OpenRouter
OPENROUTER_API_KEY="sk-or-v1-..." \
  python -m benchmarks.sycophancy.run_flipflop \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --questions 8 --rounds 4 \
  --out runs/sycophancy_gpt4o_mini.json

# Direct OpenAI
OPENAI_API_KEY="sk-..." \
  python -m benchmarks.sycophancy.run_flipflop \
  --model gpt-4o-mini --questions 8 --rounds 4
```

## Batch script (for running everything)

Save as `run_all.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "openai/gpt-4o-mini"
  "openai/gpt-4o"
  "anthropic/claude-3-haiku"
  "google/gemini-2.0-flash-001"
)

for model in "${MODELS[@]}"; do
  name="${model//\//_}"
  echo "=== Context rot: $model ==="
  python -m benchmarks.context_rot.run \
    --model "$model" \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --turns 200 --facts 10 \
    --out "runs/context_rot_${name}.json"

  echo "=== Sycophancy: $model ==="
  python -m benchmarks.sycophancy.run_flipflop \
    --model "$model" \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --questions 8 --rounds 4 \
    --out "runs/sycophancy_${name}.json"
done

echo "=== Token cost table ==="
python -m benchmarks.token_cost.model
echo "=== All done. Results in runs/ ==="
```

## Interpreting results

Every benchmark reports:

1. **Accuracy table** — two conditions (`raw` vs `memory`), with correct/total/accuracy per condition
2. **The gap** — `memory - raw` in percentage points (positive = memory recovers accuracy)
3. **Token savings** — how much smaller the memory condition's input is vs the raw condition
4. **Full JSON** — written to `runs/` for later aggregation

The mock model numbers (30% raw, 100% memory for context-rot; 100% vs 0% for sycophancy) are a methodology demonstration. Real model numbers will be different — and they'll be the actual result.

## Filling in results.md

Once you have runs for 3+ models, copy the tables into `docs/results.md` and open a PR. That turns the pitch into a published result.