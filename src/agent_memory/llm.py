"""Model client layer.

``Client`` is a tiny protocol so the whole library works against any backend.
``OpenAICompatClient`` talks to any OpenAI-compatible chat-completions endpoint
(OpenAI, together, vLLM, llama.cpp, ...). ``MockModel`` is a deterministic
emulator of the *documented* LLM failure modes (context rot, lost-in-the-middle,
sycophantic flipping) so the benchmark harnesses run fully offline in CI while
real models plug in with zero code changes.
"""

from __future__ import annotations

import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

# A message is {"role": ..., "content": ...}
Message = Dict[str, str]


class Client(Protocol):
    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        ...


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "openai package not installed; run `pip install 'agent-memory[llm]'`"
            ) from exc
        self.model = model
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **client_kwargs,
        )

    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        temperature = kwargs.pop("temperature", 0)
        model = kwargs.pop("model", self.model)
        resp = self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content or ""

    def complete_scored(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        """Like ``complete`` but also returns per-token logprobs when the server
        provides them (vLLM, OpenAI; OpenRouter for some providers).

        Returns a ``score_record``: ``text``, ``token_logprobs``, ``mean_logprob``,
        ``seq_confidence`` (= exp(mean logprob)), ``first_token_top`` (dict).
        """
        temperature = kwargs.pop("temperature", 0)
        model = kwargs.pop("model", self.model)
        top = kwargs.pop("top_logprobs", 5)
        resp = self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            logprobs=True, top_logprobs=top, **kwargs,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        token_logprobs: List[float] = []
        first_top: Dict[str, float] = {}
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if content:
            token_logprobs = [t.logprob for t in content if getattr(t, "logprob", None) is not None]
            head = getattr(content[0], "top_logprobs", None) or []
            first_top = {t.token: t.logprob for t in head}
        return score_record(text, token_logprobs, first_top)


def score_record(
    text: str,
    token_logprobs: Sequence[float] = (),
    first_token_top: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Uniform 'scored completion' record shared by every backend.

    ``seq_confidence`` is exp(mean token logprob) of the reply — the model's
    average per-token certainty in what it said. Tracking it across push-back
    rounds gives the calibration-drift signal (does confidence erode before the
    answer flips?) that plain text accuracy cannot show.
    """
    lps = [float(x) for x in token_logprobs if x is not None]
    mean_lp = sum(lps) / len(lps) if lps else None
    return {
        "text": text,
        "token_logprobs": lps,
        "mean_logprob": mean_lp,
        "seq_confidence": math.exp(mean_lp) if mean_lp is not None else None,
        "first_token_top": dict(first_token_top or {}),
    }


class LocalHFClient:
    """Run an open-weights chat model in-process with ``transformers``.

    The no-server fallback for notebooks (Kaggle T4 x2, Colab) where installing
    vLLM fights the preinstalled torch. Slower than vLLM but zero setup:
    ``device_map="auto"`` spreads a 7-8B fp16 model across two 16 GB cards, and
    token logprobs come for free from ``generate(output_scores=True)``.

    GPU-only by nature; exercised on hardware, not in CI. Use ``dtype="float16"``
    on Turing (T4) — it has no bfloat16.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        **load_kwargs: Any,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("LocalHFClient needs `pip install torch transformers accelerate`") from exc
        self._torch = torch
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device_map, **load_kwargs
        )
        self.model.eval()
        self.last_scored: Dict[str, Any] = {}

    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        return self.complete_scored(messages, **kwargs)["text"]

    def complete_scored(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - GPU path
        torch = self._torch
        temperature = float(kwargs.pop("temperature", 0) or 0)
        max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", self.max_new_tokens)))
        kwargs.pop("model", None)
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            output_scores=True, return_dict_in_generate=True, pad_token_id=pad_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        new_ids = out.sequences[0, input_ids.shape[1]:].tolist()
        token_logprobs: List[float] = []
        first_top: Dict[str, float] = {}
        for i, (step_scores, tok) in enumerate(zip(out.scores, new_ids)):
            logp = torch.log_softmax(step_scores[0].float(), dim=-1)
            token_logprobs.append(float(logp[tok].item()))
            if i == 0:
                top = torch.topk(logp, k=min(5, logp.shape[-1]))
                first_top = {
                    self.tokenizer.decode([int(t)]): float(v) for v, t in zip(top.values.tolist(), top.indices.tolist())
                }
            if tok == self.tokenizer.eos_token_id:
                break
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        self.last_scored = score_record(text, token_logprobs, first_top)
        return self.last_scored


# ---------------------------------------------------------------------------
# Mock model — deterministic emulation of documented failure modes
# ---------------------------------------------------------------------------

FACT_MARKER = re.compile(r"\[\[FACT:(\d+)\]\]\s*=\s*([^\n\]]+)")


class MockModel:
    """Scriptable model that emulates documented LLM failure modes.

    ``mode="context_rot"`` emulates lost-in-the-middle + length rot:

    * The model "retrieves" a fact only if it is near the start or end of the
      context (U-shaped attention) and the total context is short enough.
    * It replies ``<VALUE>`` when it retrieves the fact, else ``UNKNOWN``.

    ``mode="sycophancy"`` emulates self-anchoring + social pressure:

    * If the prompt includes the model's own prior answer being contradicted by
      the user (full-history condition), it flips to the user's stated value,
      with flip probability growing with the number of push-backs.
    * If the context contains no prior assistant turns (compact-memory
      condition), it answers correctly and does not flip.

    The real-model path replaces ``MockModel`` with ``OpenAICompatClient`` and
    the same prompt templates; scoring stays identical.
    """

    def __init__(self, mode: str = "context_rot", seed: int = 0) -> None:
        self.mode = mode
        self.rng = random.Random(seed)
        self.last_stats: Dict[str, Any] = {}

    def complete(self, messages: List[Message], **kwargs: Any) -> str:
        if self.mode == "context_rot":
            return self._context_rot(messages)
        if self.mode == "sycophancy":
            return self._sycophancy(messages)
        raise ValueError(f"unknown mock mode {self.mode!r}")

    # -- context rot --------------------------------------------------------

    def _context_rot(self, messages: List[Message]) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        question = _extract_question(prompt)
        if question is None:
            return "UNKNOWN"

        facts = list(FACT_MARKER.finditer(prompt))
        matches = [f for f in facts if f.group(1) == str(question["fact_id"])]
        if not matches:
            return "UNKNOWN"

        # Position of the fact's first occurrence among all fact markers.
        pos = facts.index(matches[0])
        n_facts = max(1, len(facts))
        relative = pos / n_facts if n_facts > 1 else 0.5
        length_tokens = len(prompt) / 4.0

        # Lost-in-the-middle is a *long-context* phenomenon: in a short context
        # (e.g. a compact memory file) every position is reachable, so the
        # positional penalty only applies once the context is long enough.
        short = length_tokens < 1500
        in_middle = (not short) and 0.15 < relative < 0.85
        # Emulate length rot: reliability decays as total context grows.
        rot_penalty = 0.0 if short else min(1.0, length_tokens / 40_000.0)

        found = (not in_middle) and (self.rng.random() > rot_penalty * 0.6)
        self.last_stats = {
            "relative_pos": round(relative, 3),
            "length_tokens": int(length_tokens),
            "in_middle": in_middle,
            "found": found,
        }
        return matches[0].group(2).strip() if found else "UNKNOWN"

    # -- sycophancy ---------------------------------------------------------

    def _sycophancy(self, messages: List[Message]) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        q = _extract_question(prompt)
        if q is None:
            return ""

        has_prior_assistant = any(m.get("role") == "assistant" for m in messages)
        # Count explicit user push-backs ("the answer is <wrong>").
        pushbacks = len(re.findall(r"\[\[PUSH:([^\]]+)\]\]", prompt))

        if not has_prior_assistant:
            # Compact-memory condition: nothing to anchor onto -> answer correctly.
            self.last_stats = {"flipped": False, "reason": "no prior assistant output"}
            return q["correct"]

        # Full-history condition: model sees its own prior answer being
        # contradicted. Flip probability grows with social pressure.
        flip_prob = 0.35 + 0.18 * pushbacks
        flipped = self.rng.random() < min(flip_prob, 0.95)
        self.last_stats = {
            "flipped": flipped,
            "pushbacks": pushbacks,
            "flip_prob": round(flip_prob, 2),
        }
        return q["wrong"] if flipped else q["correct"]


# ---------------------------------------------------------------------------
# Prompt helpers used by both mock and real paths
# ---------------------------------------------------------------------------

def _extract_question(prompt: str) -> Optional[Dict[str, str]]:
    """Parse a ``QUESTION <id> ...`` block carrying correct/wrong answers.

    Real models see the same template; the mock reads the annotations directly.
    """
    m = re.search(r"\[\[QUESTION:(\d+)\]\]", prompt)
    if not m:
        return None
    qid = m.group(1)
    correct = re.search(r"\[\[CORRECT:([^\]]+)\]\]", prompt)
    wrong = re.search(r"\[\[WRONG:([^\]]+)\]\]", prompt)
    if not correct or not wrong:
        return None
    return {
        "fact_id": qid,
        "correct": correct.group(1).strip(),
        "wrong": wrong.group(1).strip(),
    }


def find_fact_values(prompt: str) -> List[Tuple[str, str]]:
    """All ``[[FACT:id]] = value`` pairs present in a prompt."""
    return [(m.group(1), m.group(2).strip()) for m in FACT_MARKER.finditer(prompt)]
