.PHONY: install dev test bench-context-rot bench-sycophancy bench-tokens bench-budget demo

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

demo:
	python examples/quickstart.py
