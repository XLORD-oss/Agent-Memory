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
python -m benchmarks.budget --tier open-frontier   # DeepSeek V4 Pro, GLM-5.3, Qwen3.8-Max, Kimi K3 ≈ $54
```

Builds the exact prompts the benchmarks send, counts tokens, prices per model.
Paste the table into credit applications — see [collaboration.md](collaboration.md).

## Open-weights models on your own GPU (Kaggle T4 x2, Colab, a lab box)

```bash
# in-process, no server (transformers; float16 on T4)
python -m benchmarks.sycophancy.run_flipflop --local --model Qwen/Qwen2.5-7B-Instruct --seeds 5 --out runs/qwen7b/syc
# or a local vLLM server, then talk to it like any OpenAI-compatible endpoint
python -m benchmarks.context_rot.run --model Qwen/Qwen2.5-7B-Instruct --base-url http://127.0.0.1:8000/v1 --api-key none --seeds 5 --out runs/qwen7b/rot200
# aggregate all seeds into tables with paired bootstrap CIs
python -m benchmarks.aggregate runs/qwen7b/syc runs/qwen7b/rot200 --md runs/qwen7b/results.md
```

`--seeds N` writes one JSON per seed and skips seeds already on disk, so a run
killed by a session limit resumes. Local backends also record per-token
logprobs → confidence drift per condition. Full walkthrough: [kaggle.md](kaggle.md).

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