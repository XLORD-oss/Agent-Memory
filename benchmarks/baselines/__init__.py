"""Baseline arms contributed by collaborators.

Each baseline is one module that registers one arm with
``benchmarks.common.arms.register_arm``. Add a module here, import it below,
and it becomes selectable as ``--arms <name>`` in the sycophancy runner and is
reported by ``benchmarks.aggregate`` alongside the built-in arms.

Attribution: put the author and the paper/method the baseline implements in
the module docstring; git history carries the rest. This is the intended
place for a collaborator's contribution to live as a distinct, citable unit.

Expected baselines (see docs/proposal.md §3):

* ``rolling_summary``  — an LLM summary of the conversation so far replaces the
                         transcript (the most common production alternative).
* ``rag``              — top-k transcript chunks retrieved for the current turn,
                         under the same token budget as the memory arm.
* ``memgpt``           — a MemGPT/Letta-style core+archival memory agent.
"""

from . import rolling_summary  # noqa: F401
