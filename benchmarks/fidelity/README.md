# Memory-fidelity benchmark — the assimilation error

Every other benchmark measures the **model's** behavior given context. This one
measures the **memory itself**: how much of what happened survives the
compaction, *before a model ever reads it*.

The framing comes from reduced-order modeling: the compact memory is a lossy
compression (distillation + dedupe + archiving) of the conversation — a
**reduced-order model of the transcript**. The questions are the ones you'd ask
of any truncation:

* **State fidelity** — what fraction of planted ground truth is in the *active*
  memory the model sees? This is the conversation's rate-distortion curve:
  memory size vs recall.
* **Archival completeness** — what's recoverable anywhere (active + `archive/`)?
  The archive is the lossless audit trail; recall_total should be ~100% always.
* **Distillation loss** — the gap between `MarkerDistiller` (perfect extraction,
  control) and `RuleDistiller` (realistic heuristic extraction).
* **Retention loss under pressure** — how usage-weighted eviction protects facts
  the model actually references, vs facts that never get used.

## Run

```bash
python -m benchmarks.fidelity.run                          # defaults
python -m benchmarks.fidelity.run --distiller rules        # realistic distillation
python -m benchmarks.fidelity.run --caps 40,80,160,320 --archive-policy oldest
python -m benchmarks.fidelity.run --plot rate_distortion.png
```

## Reading the output

```
| cap | recall active | recall archive | recall total | lost | precision | entries act/all | distinct facts | mem tok | comp |
|  40 | 21% | 100% | 100% | 0% | 100% | 5/46 | 5 | 35 | 59.5x |
|  80 | 50% | 100% | 100% | 0% | 100% | 12/38 | 12 | 80 | 26.0x |
```

* `recall active` = **state fidelity** — what the model can see. This is the
  headline. It drops when the cap is too small for the signal.
* `recall archive` = how much of that loss the archive still covers.
* `recall total` = archival completeness; `lost` = truly erased (should be 0% by
  design — archiving never deletes).
* `precision` = share of active entries that are real facts (marker control is
  100%; `RuleDistiller` adds conclusions/preferences, so it drops).

Then the source-of-loss breakdown and the usage-protection comparison:

```
Source-of-loss breakdown at cap = 40:
  distillation loss (marker − rules state fidelity):  +21% (perfect=21%, rules=0%)
  staleness (archive covers active gaps, marker):     79% (recall_total=100%)

Usage-weighted protection at cap=160 (rules distiller):
  state fidelity — referenced facts: 50% vs unreferenced: 8% (+42% protection)
  vs no usage tracking: referenced state fidelity 42% (+8% gain)
```

## Methodology (rigor notes)

1. **Plant** N facts with distinct values, duplicate each R times (tests dedupe),
   echo a subset in later assistant answers (tests usage tracking), bury
   everything in distractors + conclusions + preferences.
2. **Replay** the scripted conversation through a fresh `MemoryEngine` under a
   chosen distiller, cap, profile, and archive policy.
3. **Score by substring on the memory files** — no model is called, so the
   measurement isolates the memory from model variance. Fully offline,
   deterministic given the seed.
4. **Two distillers isolate the two losses**: `MarkerDistiller` extracts the
   planted markers verbatim (the compression ceiling); `RuleDistiller` is the
   realistic path (patterns only) and also keeps conclusions/preferences.

## Honest caveats

* Recall is exact-value substring match (planted values are single distinct
  words). Real paraphrased memories would need an LLM judge — this is the
  *floor* estimate of fidelity, which is what makes a positive result
  meaningful.
* More facts / tighter caps → sharper curves. Defaults (24 facts, caps
  40–240) are tuned so the interesting regime (state fidelity 20%→100%) shows
  on the default run.
* `usage` protection shows through the realistic `RuleDistiller` (entry texts
  must be long enough for the token-overlap tracker to fire). Tiny marker
  entries are below the tracker's threshold — a documented limitation of the
  zero-cost usage heuristic, not of the protection mechanism.