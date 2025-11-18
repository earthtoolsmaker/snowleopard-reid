.PHONY: help install lint format check fiftyone-load fiftyone-export

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies with uv sync
	uv sync

lint: ## Run ruff linter
	uv run ruff check .

format: ## Run ruff formatter
	uv run ruff format .

check: ## Run ruff check and format in check mode (no changes)
	uv run ruff check .
	uv run ruff format --check .

fiftyone-load: ## Load YOLO dataset into FiftyOne and launch app for review
	uv run python scripts/data/load_yolo_to_fiftyone.py --overwrite --launch

fiftyone-export: ## Export samples tagged with 'selected' from FiftyOne to JSON
	uv run python scripts/data/export_fiftyone_selection.py --selection-mode selected
