.PHONY: install dev test bench-context-rot bench-sycophancy bench-tokens bench-budget export-demo demo

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest

bench-tokens:
	python -m benchmarks.token_cost.model

bench-context-rot:
	python -m benchmarks.context_rot.run --mock

bench-sycophancy:
	python -m benchmarks.sycophancy.run_flipflop --mock

bench-budget:
	python -m benchmarks.budget

export-demo:
	python -m agent_memory.export --demo --out /tmp/agent-memory-sft

demo:
	python examples/quickstart.py
