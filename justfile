set shell := ["bash", "-cu"]

APP := "main.py"

# Default: show commands
default:
    just --list

# Create venv (uv-managed)
venv:
    uv venv

# Install dependencies
install:
    uv pip install -r requirements.txt

# Initialize projetct
init:
    uv init

# sync the environement
sync:
    uv sync

# Run the app
run:
    uv run {{APP}}

# Run with args: just run -- --debug
run-args args:
    uv run {{APP}} {{args}}

# Tests
test:
    uv run pytest

# Check
check:
    uv tool run ruff check

# Lint
lint:
    uv tool run ruff format

# Update deps
upgrade:
    uv add -Ur requirements.txt

# Clean caches
clean:
    rm -rf .venv __pycache__ .pytest_cache
