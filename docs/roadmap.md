# Roadmap

The ordering is deliberate: **the benchmarks come before the integrations.** The
framework's value stands or falls on two numbers — the context-rot gap and the
sycophancy gap — and nothing is worth building on top until those are measured
with real models.

## v0.1 — this repository (done)

- [x] Compact memory engine: ingest → distill → merge → archive → build_context
- [x] Human-readable `memory.md` / `perspectives.md` + archive split
- [x] Rules-based and LLM-based distillers
- [x] Token-cost model (O(T²) vs O(T))
- [x] Context-rot benchmark (Chroma methodology) — runs offline with mock
- [x] Sycophancy FlipFlop benchmark (SYCON ToF/NoF metrics) — runs offline with mock
- [x] Unified memory policy (priority × usage × recency × affinity per kind)
- [x] Usage-weighted retention + pinning + auto task detection
- [x] **`analyze_trace`** — replay real chat logs through any policy, offline;
      per-turn token curve, usage report, profile comparison, plots
- [x] Unit + harness tests, CI, packaging

## v0.2 — make the numbers real

- [ ] Run both benchmarks on ≥5 frontier models (Claude, GPT, Gemini, Qwen, Llama),
      same script, publish the tables in `docs/results.md`
- [ ] LLM-judge scoring alongside exact-match (paraphrase-tolerant correctness)
- [ ] Distractor-injection test: confirm that memory is robust to *semantically
      similar* wrong facts (the Chroma finding that distractors mislead models)
- [ ] Robust `LLMDistiller` with schema validation + retry; measure distillation
      fidelity (how much of the ground truth survives into memory)
- [ ] Archive-policy ablation: cap size, recency vs importance vs kind priority

## v0.3 — production integrations

- [ ] Adapters: LangChain / LlamaIndex / OpenAI Agents SDK memory drop-in
- [ ] MCP (Model Context Protocol) memory server so any agent can use it
- [ ] Embedding-assisted dedupe/merge (upgrade from token-overlap heuristic)
- [ ] Structured distillation with typed facts (dates, entities, preferences)
- [ ] Memory diffing + human correction loop (edit the file, engine re-syncs)

## v1.0 — the published result

- [ ] A writeup: *context-rot gap*, *sycophancy gap*, *cost ratio* on real models
- [ ] The SACD-specific study: does compact-memory replay reduce self-anchoring
      calibration drift? (New result — nobody has tested this with compact replay)
- [ ] Multi-session memory: profile persists across chats, facts dedupe globally
- [ ] PII / security handling for long-lived memory stores

## Research agenda (open questions worth testing)

1. **Sycophancy magnitude.** FlipFlop with full history vs memory file, controlled
   for model and task. The repo ships the harness; the result is the deliverable.
2. **Distractor immunity.** Does a compact memory with one wrong-but-plausible
   entry poison answers less than a long transcript with the same wrong entry?
3. **Archive placement.** Where entries land (recently-active vs topic-split)
   measurably changes retrieval; worth an ablation.
4. **Distillation fidelity.** How much durable information is lost in distill?
   Tradeoff curve: memory size vs recall vs cost.

## Non-goals (for now)

* Not a general-purpose vector database / RAG product.
* Not a model-training or fine-tuning layer.
* Not a replacement for tool-calling or planning frameworks.
