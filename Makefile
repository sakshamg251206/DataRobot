.DEFAULT_GOAL := help
PY ?= python

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install the app with ML extras and dev tools (editable)
	$(PY) -m pip install -e ".[ml,dev]"

run: ## Start the app at http://localhost:8501
	streamlit run app.py

lint: ## Lint and check formatting
	ruff check .
	ruff format --check .

format: ## Auto-format and fix lint issues
	ruff format .
	ruff check --fix .

typecheck: ## Static type checks
	mypy

test: ## Run the test suite
	pytest

coverage: ## Run tests with a coverage report for the core logic
	pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

docker: ## Build and run the Docker image
	docker build -t autods .
	docker run --rm -p 8501:8501 --env-file .env.example autods

.PHONY: help install run lint format typecheck test coverage check docker
