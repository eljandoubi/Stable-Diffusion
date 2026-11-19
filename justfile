set shell := ["bash", "-cu"]

# Default: show commands
default:
    just --list

# Sync the environment
sync:
    uv sync

# Tests
test:
    uv run pytest

# Lint
lint:
    uv run ruff check .

# Format
format:
    uv run ruff format .

# Lint, format check and tests (what CI runs)
ci:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

# Clean caches
clean:
    rm -rf .venv __pycache__ .pytest_cache .ruff_cache
