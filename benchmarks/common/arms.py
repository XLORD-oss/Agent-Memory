"""Registry of sycophancy-benchmark arms (conditions).

An *arm* is a function ``(client, item, rounds, system) -> row`` that runs one
item through one context contract and returns the standard row::

    {"tof": int|None, "nof": int, "initial": str, "initial_correct": bool,
     "confidence": [float|None, ...]}   # round 0 .. round R

Built-in arms live in ``benchmarks/sycophancy/run_flipflop.py``. **Baselines
contributed by collaborators live in ``benchmarks/baselines/``** and register
themselves here, so a contribution is a separate, attributable module rather
than an edit to the core runner::

    from benchmarks.common.arms import register_arm

    @register_arm("rolling_summary")
    def run_item_rolling_summary(client, item, rounds, system) -> dict:
        ...

Once registered, the arm is selectable with ``--arms rolling_summary`` and is
aggregated like any other (``benchmarks.aggregate`` reports every arm present).
"""

from __future__ import annotations

from typing import Callable, Dict, List

Arm = Callable[..., dict]

_REGISTRY: Dict[str, Arm] = {}


def register_arm(name: str) -> Callable[[Arm], Arm]:
    """Decorator: make ``fn`` selectable as ``--arms <name>``."""

    def deco(fn: Arm) -> Arm:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"arm {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def get_arm(name: str) -> Arm:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(_REGISTRY)}") from None


def available_arms() -> List[str]:
    return sorted(_REGISTRY)


def row(a0: str, correct: bool, flips: List[int], confidences: List) -> dict:
    """Build the standard result row (shared by built-in arms and baselines)."""
    return {
        "tof": flips[0] if flips else None,
        "nof": len(flips),
        "initial": a0,
        "initial_correct": correct,
        "confidence": confidences,
    }
