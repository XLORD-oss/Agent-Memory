.PHONY: install dev docs docs-serve docs-check test bench-context-rot bench-sycophancy bench-tokens bench-budget bench-aggregate export-demo demo

install:
	pip install -e .

dev:
	pip install -e ".[dev,docs]"

docs:
	python scripts/build_docs.py

docs-serve:
	python scripts/build_docs.py --serve

docs-check:
	python scripts/build_docs.py --check

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

bench-aggregate:
	python -m benchmarks.aggregate runs/sycophancy_flipflop runs/context_rot

export-demo:
	python -m agent_memory.export --demo --out /tmp/agent-memory-sft

demo:
	python examples/quickstart.py
