.PHONY: help install lint format check catalog-viewer

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

catalog-viewer: ## Launch the snow leopard catalog viewer UI
	uv run python scripts/ui/leopard_catalog_viewer.py --catalog-root data/08_catalog/v1.0 --port 7860
