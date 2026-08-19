# medrl task runner. `just --list` to see everything.

default: --list
set shell := ["bash", "-cu"]

venv := ".venv/bin"
python := venv + "python"

# install dev + core (CPU-safe)
install:
    uv pip install -e ".[dev]"

# CPU-only unit tests
test:
    {{python}} -m pytest -q

# lint + strict types
check:
    {{venv}}/ruff check src tests
    {{venv}}/mypy src/medrl

# everything CI runs
ci: check test

# assert Qwen3.5 architecture invariants against the live Hub (network)
probe repo="Qwen/Qwen3.5-9B":
    {{python}} -m medrl.model.probe --repo "{{repo}}"

# list the benchmark registry
tasks:
    {{venv}}/medrl eval tasks

# validate a config and show its deployment plan (CPU-safe)
plan preset="decision_grade":
    {{venv}}/medrl eval plan -c "{{preset}}"

# emit the text-only checkpoint (GPU host, [train] extra)
convert source="Qwen/Qwen3.5-9B" out="/scratch/medrl/checkpoints/qwen35-9b-text":
    {{venv}}/medrl model convert --source "{{source}}" --out "{{out}}"
