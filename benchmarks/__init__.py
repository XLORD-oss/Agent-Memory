"""Benchmark package for Agent-Memory.

Each subpackage is a self-contained experiment built on a published methodology:

* ``context_rot``  — Chroma's focused-vs-full-prompt methodology, adapted to
  compare a full raw transcript against the compact memory file.
* ``sycophancy``   — the FlipFlop / SYCON "push back until it flips" setup,
  run once with full history and once with the memory file.
* ``token_cost``   — the O(T²) vs O(T) arithmetic, no model needed.
"""
