# Single-command entry points. CI calls these exact targets so that "it passes locally"
# and "it passes in CI" cannot mean different things.
#
# On Windows without make, every target below is a thin wrapper: run the `arag-eval ...`
# or `pytest ...` command directly.

PY ?= python

.PHONY: help install validate eval eval-fast record report judge-agreement test lint typecheck cov ci

help:
	@echo "install         install the package plus dev extras"
	@echo "validate        validate the golden set and report composition shortfall"
	@echo "eval-fast       stratified 20-item subset, cassette replay, no network  [PR GATE]"
	@echo "eval            full suite, live models, records a run                  [NIGHTLY]"
	@echo "record          re-record provider cassettes (costs API calls)"
	@echo "report          regenerate EVAL_LOG.md from data/eval_runs.jsonl"
	@echo "judge-agreement Cohen's kappa, local judge vs human labels"
	@echo "test lint typecheck cov"

install:
	$(PY) -m pip install -e ".[dev]"

validate:
	arag-eval validate

# The PR gate. Deterministic, free, offline. Exit 1 = quality regression, 2 = broken harness.
eval-fast:
	arag-eval run --engine null --fast

# The nightly. Live models, full set, thresholds enforced, run appended to the log.
eval:
	arag-eval run --engine null --concurrency 2

record:
	arag-eval run --engine null --record --no-log

report:
	arag-eval report

judge-agreement:
	arag-eval judge-agreement data/judge_labels.jsonl

test:
	pytest -m "not live"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

cov:
	pytest -m "not live" --cov --cov-report=term-missing --cov-fail-under=70

# What CI runs on a pull request.
ci: lint typecheck cov eval-fast
