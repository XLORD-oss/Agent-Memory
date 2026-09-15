# Which models to test — and what "open source" buys you for each

"Open weights" and "runnable on your hardware" are different properties. The
models you named are all open-weight (or promised); none of them is runnable on
a Kaggle T4 ×2. That is not a problem — it just puts them in a different lane
of the study.

## The four you named, checked

| Model | Size | Weights | Self-host reality | API price (in/out per M) |
|---|---|---|---|---|
| **Kimi K3** (Moonshot) | 2.8 T total / 104 B active MoE, 1 M context | shipped 27 Jul 2026 (Kimi K3 License; MXFP4 checkpoint) | multi-node H100/MI300, ≥ 8 GPUs | $3 / $15 |
| **DeepSeek V4 Pro** | 1.6 T total / 49 B active MoE, 1 M context | MIT, live since Apr 2026 | 8×H100-class node | $1.32 / $3.96 peak, half off-peak |
| **GLM-5.3** (Z.ai) | ~743 B total / ~40 B active (same base as 5.2, more RL) | API-first (14 Aug 2026); weights promised ~2 weeks later — **verify on HF before citing as open** | 8×H100 / 4×MI300 | ~$1.40 / $4.40 |
| **Qwen3.8-Max** | 2.4 T / 95 B active, multimodal | Max itself is API-only; the open-weight variant is the text-only `Qwen3.8-2.4T-A95B`; **`Qwen3.8-27B` dense, Apache 2.0** shipped 14 Aug 2026 | Max: 8-GPU class. 27B: one 24–48 GB card, or T4×2 in 4-bit | Max $2 / $6; 27B ≈ $0.20 / $0.60 |

Sources: model cards and launch coverage as of Sept 2026; prices are list
prices and drift — `benchmarks/budget.py --tier open-frontier` has the numbers
used here and takes `--price` overrides.

Two things follow:

1. **For the frontier four, "open weights" changes the license and the
   audit trail, not the compute.** You will call them through an API exactly
   like GPT-4o. The honest label in the paper is "open-weight models via
   hosted inference."
2. **The self-hosting win in your list is Qwen3.8-27B**, not Max. A dense 27B
   with Apache 2.0 that fits on two T4s in 4-bit (or one card at ~14 GB) gives
   you the strongest model you can run *with logprobs*, which the API lane
   mostly cannot (roughly a quarter of OpenRouter endpoints return logprobs at
   all, and top-k only).

## The study's model ladder

Three lanes, each with a specific job. Do not collapse them.

| Lane | Models | What it answers | Logprobs | Cost |
|---|---|---|---|---|
| **A. Local** (Kaggle T4 ×2) | Qwen2.5-7B, Llama-3.1-8B, **Qwen3.8-27B (4-bit)** | Does the contract effect exist? Confidence-drift (self-anchoring) figure. 5–10 seeds cheaply. | ✅ full | $0 |
| **B. Open-weight frontier** (API) | DeepSeek V4 Pro, GLM-5.3, Qwen3.8-Max, Kimi K3 | Does the effect **survive scale**? Reviewers' first objection to lane A is "small models are sycophantic anyway." | partial/none | ≈ $54 for the whole 5-seed campaign |
| **C. Closed frontier** (API) | GPT-4o, Claude 3.5 Sonnet, Gemini Flash | Same question, other training pipelines; the models practitioners actually deploy. | mostly none | ≈ $43 |

Budgets from `python -m benchmarks.budget --tier <lane>`; > 95 % of every
lane's cost is the *raw full-transcript* baseline.

### Why lane B is worth its $54

The strongest pre-emptive objection to the whole project is that
sycophancy and lost-in-the-middle are *small-model* pathologies that frontier
training has already fixed. Chroma's context-rot result (18 models including
GPT-4.1 and Claude 4) says no for context rot; for sycophancy the picture is
mixed — reasoning-tuned and larger models resist better, and Kimi/DeepSeek/GLM
are all heavily RL-post-trained. So lane B can go either way, and *that* is
the finding: either the contract still buys something at 1–3 T parameters
(strong paper) or the gap shrinks with scale (still a paper — a scaling curve
of the sycophancy gap across three lanes is a figure nobody has published).

### What lane B cannot give you

* **Confidence drift.** No logprobs → no calibration story. That stays in lane A
  and is the reason the 27B matters: it is the largest model you will have
  logprobs for.
* **Thinking modes.** DeepSeek V4 Pro and Qwen3.8-Max have hybrid reasoning.
  Run them with thinking **off** for the main table (the harness asks for
  one-value answers; a 30-second reasoning trace per call changes both cost
  and behaviour) and report thinking-on as an ablation if budget allows.
* **Stability.** Hosted open-weight models get re-routed between providers
  with different quantisations. Pin the provider on OpenRouter where possible
  and record `response.model` — the harness writes `args.model`, so add the
  provider to the run tag.

## Concrete commands

```bash
# Lane A — Qwen3.8-27B on Kaggle T4 x2, 4-bit, in-process
pip install -q bitsandbytes accelerate
python -m benchmarks.sycophancy.run_flipflop --local --model Qwen/Qwen3.8-27B \
    --dtype float16 --seeds 5 --out runs/qwen38-27b/syc          # add load_in_4bit via LocalHFClient(**load_kwargs) if it does not fit fp16 (it won't: 27B fp16 ≈ 54 GB)

# Lane B — the frontier four via OpenRouter (no thinking)
for M in deepseek/deepseek-v4-pro z-ai/glm-5.3 qwen/qwen3.8-max moonshotai/kimi-k3; do
  TAG=${M//\//_}
  python -m benchmarks.sycophancy.run_flipflop --model $M --base-url https://openrouter.ai/api/v1 \
      --api-key-env OPENROUTER_API_KEY --seeds 5 --out runs/$TAG/syc
  python -m benchmarks.context_rot.run --model $M --base-url https://openrouter.ai/api/v1 \
      --api-key-env OPENROUTER_API_KEY --turns 1000 --facts 10 --seeds 5 --out runs/$TAG/rot1000
done
python -m benchmarks.aggregate runs/*/syc --md results_syc.md
```

For the 27B in 4-bit, `LocalHFClient` passes extra keyword arguments straight to
`from_pretrained`, so a notebook can do:

```python
from agent_memory.llm import LocalHFClient
from transformers import BitsAndBytesConfig
client = LocalHFClient("Qwen/Qwen3.8-27B", dtype="float16",
                       quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                              bnb_4bit_compute_dtype="float16"))
```

## What not to do

* Don't call lane B "open source" in a paper without the license per model
  (K3 has its own license; DeepSeek is MIT; GLM-5.3's terms were unpublished
  at launch; Qwen3.8-Max is closed, its 2.4T sibling and the 27B are open).
* Don't compare thinking-on frontier models against thinking-off small models
  and call the difference "scale".
* Don't spend lane B money before lane A shows a gap. If a 7B model with the
  strongest possible failure modes shows nothing, the $54 buys you nothing but
  a cleaner null.
