# Repository Guidelines

## Project Structure & Module Organization
This repository is a small PyTorch codebase for Intertwined H-JEPA experiments. Core model logic lives in `intertwined_hjepa.py`, training and checkpoint inspection live in `scripts/`, dataset helpers live in `data/`, and tests live in `test/` plus `test_dataset_helpers.py`. Keep new code close to the subsystem it belongs to; avoid introducing broad framework layers.

## Build, Test, and Development Commands
Use `uv` for all environment and run commands.

- `uv sync` installs dependencies and creates the local `.venv`.
- `uv run -- pytest test test_dataset_helpers.py` runs the full test set.
- `uv run -- pytest test/test_intertwined_hjepa_shapes.py test/test_intertwined_hjepa_training_step.py` runs model-focused tests.
- `uv run -- python -m scripts.train_intertwined_hjepa ...` starts training.
- `uv run -- python -m scripts.inspect_checkpoint ...` inspects checkpoints and emits diagnostics.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes. Keep tensor shapes explicit in names and comments when they are non-obvious. Prefer short, direct helpers over abstraction-heavy wrappers. This repo uses PyTorch, pytest, and `uv`; keep new code compatible with that workflow. The current stack targets Blackwell with PyTorch `2.9.1`, CUDA `13.0` wheels, `torchao`, and `flash-attn-4==4.0.0b9`.

## Testing Guidelines
Use pytest. Test files live under `test/` and follow `test_*.py` naming. Add a focused smoke test for any model or trainer change, especially if it touches shapes, EMA updates, optimizer behavior, or checkpointing. Prefer tiny synthetic inputs when possible so tests stay fast and deterministic.

## Commit & Pull Request Guidelines
Use imperative, behavior-specific commit messages such as `fix JEPA sequence shape handling`. Keep commits scoped to one concern when possible. Pull requests should summarize the change, explain why it is needed, and list the tests run. Include screenshots only for docs, figures, or generated outputs.

## Dependency & Environment Notes
Before adding a dependency, verify it is necessary and maintained. Avoid manual `venv` or `pip` workflows in normal use; this repo is set up around `uv sync`, `uv lock`, and `uv run`.
