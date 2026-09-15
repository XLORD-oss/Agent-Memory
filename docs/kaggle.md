# Running the real-model pilot on Kaggle (T4 ×2, 30 h/week, 9 h/session)

**Verdict first.** Kaggle's T4 ×2 is *enough for the whole inference pilot* —
both benchmarks, several open models, 5+ seeds, with token logprobs (which the
paid APIs mostly won't give you). It is *not* enough for the H4 fine-tuning
study: one cell of the 2×2 fits, four cells × three seeds does not. So the
split is: **pilot here, training with a university.** Do the pilot first — it
is what makes the training ask credible.

## What a T4 ×2 actually is

Two 16 GB Turing cards (compute capability 7.5), ~320 GB/s each, **no
bfloat16** — everything runs in `float16`. That determines the model list:

| Model | Fits how | Use for |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | fp16 across both cards (~15 GB) | primary pilot model |
| `meta-llama/Llama-3.1-8B-Instruct` | fp16 across both cards (~16 GB, tight) or AWQ 4-bit on one | second family |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` / `-GPTQ-Int4` | 4-bit on one card | "does the gap shrink with scale?" |
| **`Qwen/Qwen3.8-27B`** (dense, Apache 2.0, Aug 2026) | 4-bit via bitsandbytes across both cards (~15 GB); fp16 does **not** fit | the strongest model you will have logprobs for — see `models.md` |
| `Qwen/Qwen2.5-1.5B-Instruct`, `-3B-Instruct` | trivially | debugging the pipeline in minutes |

Anything above ~30B is not realistic here — the trillion-parameter open-weight
MoEs (Kimi K3, DeepSeek V4 Pro, GLM-5.3, Qwen3.8-2.4T) need 8×H100-class nodes
and belong in the API lane (`models.md`). That is fine: the hypothesis is about
the *contract*, and a 7B family already shows sycophancy and lost-in-the-middle
strongly (small models show them *more*, which is a limitation to state, not a
reason to skip).

## Two ways to serve the model

**Path A — vLLM server (fast, preferred).** vLLM supports compute capability
≥ 7.0, so the T4 runs it — but it must be forced to `--dtype float16`
(bf16 raises `ValueError` on capability < 8.0), and you must not be greedy
with VRAM (`--gpu-memory-utilization 0.90`). The harness then talks to it as
an ordinary OpenAI-compatible endpoint and gets logprobs for free.

```bash
pip install -q vllm                                   # may take ~5 min; if it fights the preinstalled torch, use Path B
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --dtype float16 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --port 8000 > vllm.log 2>&1 &
# wait until vllm.log says "Uvicorn running"
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=none
```

**Path B — in-process (`--local`, zero setup).** Uses `transformers` with
`device_map="auto"` to spread the model over both cards, and reads logprobs
from `generate(output_scores=True)`. 3–10× slower than vLLM, no server to
babysit, works when the vLLM install fails.

```bash
pip install -q -e ".[dev]" accelerate
python -m benchmarks.sycophancy.run_flipflop --local --model Qwen/Qwen2.5-7B-Instruct --dtype float16 ...
```

## The campaign, sized for 9-hour sessions

Every runner takes `--seeds N` and writes `runs/<name>/seed_k.json`, **skipping
seeds already on disk**. A session that dies at 9 h loses at most the seed in
flight; rerun the same command and it resumes. Save `runs/` as a Kaggle
Dataset output at the end of each session (or `git add runs/…` on a scratch
branch) so it survives the kernel reset.

```bash
git clone -b arena/01a02e08-agent-memory https://github.com/XLORD-oss/Agent-Memory.git
cd Agent-Memory && pip install -q -e ".[dev]"

M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b            # (Path A: vLLM running; Path B: add --local)

# 1. sycophancy — 8 items × 4 rounds × 2 conditions ≈ 80 calls/seed; minutes per seed
python -m benchmarks.sycophancy.run_flipflop --model $M --seeds 5 --out runs/$TAG/sycophancy

# 2. context rot — raw condition sends the whole transcript; keep under the model's window
python -m benchmarks.context_rot.run --model $M --turns 200  --facts 10 --seeds 5 --out runs/$TAG/rot200
python -m benchmarks.context_rot.run --model $M --turns 600  --facts 10 --seeds 5 --out runs/$TAG/rot600   # ≈13k tokens/prompt

# 3. tables with paired bootstrap CIs
python -m benchmarks.aggregate runs/$TAG/sycophancy runs/$TAG/rot200 runs/$TAG/rot600 --md runs/$TAG/results.md
```

Rough budget (Path A; multiply by ~5 for Path B):

| Step | Calls | Prompt size | Time on T4 ×2 |
|---|---|---|---|
| sycophancy, 5 seeds | 400 | 200–350 tok | ~10–20 min |
| rot200, 5 seeds | 100 | raw ≈ 4.6k tok, memory ≈ 0.5k | ~10 min |
| rot600, 5 seeds | 100 | raw ≈ 13k tok, memory ≈ 0.5k | ~30–45 min (prefill-bound) |
| second model family | | | same again |
| **Two models, everything** | | | **≈ 2–4 h of the 30 h/week** |

The whole pilot fits in **one session** per model. That leaves 20+ hours a week
for the ablations: `--rounds 8`, the 14B AWQ scale point, a `--system neutral`
variant (the memory contract's system prompt carries anti-sycophancy
instructions — run both conditions with a neutral prompt too, otherwise the
prompt, not the memory, could be the cause).

## What the T4 gives you that the APIs don't: confidence

Every scored run records per-round `seq_confidence` = exp(mean token logprob)
of the reply. `aggregate` reports **confidence drift** (round 0 → last round)
per condition. That is the self-anchoring measurement: does the model's
certainty erode under push-back *before* it flips, and does the erosion differ
when it cannot see its own prior answer? Text-only APIs cannot show this. It is
the one figure in the pilot note that a reviewer from the calibration
literature will stop on.

## What this does *not* cover — and where the university comes in

| | Kaggle T4 ×2 | Needs a cluster |
|---|---|---|
| Inference pilot, 7–14B, 5+ seeds, logprobs | ✅ | |
| LLM distillation pass (`LLMDistiller` with a local 7B) | ✅ slow but fine | |
| One QLoRA cell of H4 (7B, 4-bit base, r=16) | ⚠️ fits in 16 GB, ~hours per epoch on ~10M tokens; would eat most of a week's quota for *one* cell | |
| Full H4: 4 cells × 3 seeds, fp16 LoRA, identical hyperparameters | ❌ | ✅ ~50–100 A100/H100-hours |
| Cross-eval matrix: 12 adapters × 4 formats with logprobs | ❌ | ✅ |
| Models ≥ 30B | ❌ | ✅ |

You are right that training assistance is close to inevitable. Two honest
qualifications:

1. **One QLoRA cell is a legitimate Kaggle experiment** — e.g. train only
   `memory` vs `full` on a small trace set and look at the sign of the effect.
   A pilot-of-the-pilot. It will not be publishable on its own (no length
   control, one seed, 4-bit base), but it tells you whether the training ask
   is worth making at all.
2. **IndiaAI's subsidised GPUs** (~₹65/GPU-hr for academic researchers) are an
   alternative to a collaborator's cluster for raw compute; they are not an
   alternative to a collaborator for design review, baselines and the arXiv
   endorsement. Read `collaboration.md` before deciding which you are asking
   for.

## Gotchas

* **fp16 overflow.** Rare with Qwen2.5/Llama-3.1 for inference; if you see
  `nan` logprobs or garbage after long prompts, drop `--max-model-len` or use
  the AWQ variant.
* **Context window.** `--turns 600` is ≈ 13k tokens per raw prompt; keep
  `--max-model-len` above that. `--turns 5000` (≈ 107k) does **not** fit —
  that regime is for the API models.
* **Gated models.** Llama needs `huggingface-cli login` with an accepted
  license; Qwen does not.
* **Kernel resets wipe `runs/`.** Push to a scratch branch or save as a Dataset
  every session; the runners resume, but only from files that still exist.
* **Do not report mock numbers.** The harness labels mock runs
  `model: "mock"`; if that string appears in a results table it is not a
  result.
