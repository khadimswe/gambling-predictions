PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup data features tiers train lint format typecheck test text secrets check pdf clean

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## create .venv, install everything, enable git hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,report]"
	$(BIN)/pre-commit install

data: ## download raw CSVs from football-data.co.uk
	$(BIN)/soccer download

features: ## build data/processed/features.csv (includes tiers)
	$(BIN)/soccer build

tiers: ## Khadim: rerun tier experiments, refresh reports/results and figures
	$(BIN)/soccer tiers-report

train: ## Ryann: train + evaluate the classifier on validation
	$(BIN)/soccer train --model logreg

lint: ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## auto-fix lint and formatting
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## mypy
	$(BIN)/mypy

test: ## pytest with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

text: ## no em dashes or invisible characters
	$(BIN)/python scripts/check_text.py

secrets: ## gitleaks over full git history (brew install gitleaks)
	gitleaks git --config .gitleaks.toml --redact .

check: lint typecheck test text secrets ## everything CI runs

pdf: ## render specs and the midterm report to build/pdf/*.pdf
	$(BIN)/python scripts/build_pdf.py docs/specs/khadim-tiers.md docs/specs/ryann-classifier.md reports/midterm/midterm-report.md

clean: ## remove caches and build output
	rm -rf build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
