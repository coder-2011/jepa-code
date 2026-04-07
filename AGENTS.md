# Repository Guidelines

Naman owns this repository.
Editor: `code <path>`.

## Project Objective
Use this workspace to study, compare, and improve JEPA-style architectures. The current focus is understanding the legacy `jepa/` package, referencing `lejepa/`, and building cleaner JEPA implementations with explicit architectural contracts.

## Stack Snapshot
- Python-first workspace.
- Primary libraries: PyTorch, NumPy, PyYAML, pytest.
- `jepa/` is a packaging-heavy training framework with docs/examples.
- `lejepa/` is a leaner research codebase centered on LeJEPA/SIGReg.

## Root Folder Structure
- `jepa/`: legacy JEPA package under active inspection.
  - `jepa/jepa/`: source package
  - `jepa/tests/`: tests
  - `jepa/docs/`: docs
  - `jepa/examples/`: examples
- `lejepa/`: reference implementation and evaluation code.
  - `lejepa/lejepa/`: source package
  - `lejepa/tests/`: tests
  - `lejepa/scripts/`: training utilities
  - `lejepa/eval/`: eval assets and outputs
- `JEPA.pdf`, `LeJEPA.pdf`: architecture references; do not commit derived artifacts unless explicitly needed.

## Edit Boundaries
- Never edit, refactor, delete, or otherwise modify files inside `jepa/`.
- Never edit, refactor, delete, or otherwise modify files inside `lejepa/`.
- Those folders are reference material only in this workspace.
- New implementation work should live outside those folders unless the user explicitly overrides this rule.

## Reasoning and Explanations
- Prioritize clarity and truthfulness.
- Separate canonical JEPA ideas from what a given repo happens to implement.
- Mark facts, assumptions, and architectural inferences clearly.
- Explain concepts simply, but go in lots of depth when architecture decisions are involved.

## Evidence-First Reasoning Mindset
- Flow: evidence -> belief update -> conclusion.
- Prefer paper references, code paths, and runnable smoke tests over README claims.
- Treat local docs as helpful but not authoritative when they conflict with code or primary papers.

## Agent Learning Log
- Use only `.codex/learning.md` for durable project learnings.
- Read it before non-trivial implementation work.
- Append concise notes for architecture decisions, shape-contract bugs, environment issues, and test pitfalls.
- Do not use `.codex/STATE.md` in this repo.

## Environment Setup
- For `jepa/`:
  - `cd jepa && pip install -e ".[dev,docs]"`
  - `cd jepa && pytest`
- For `lejepa/`:
  - `cd lejepa && pip install -e .`
  - `cd lejepa && pytest`
- Prefer project-local virtual environments such as `.venv/`.
- Keep setup commands aligned with `README.md`, `pyproject.toml`, and this file when workflow changes.

## Commit and PR Rules
- Work in atomic, scoped commits.
- Keep changes isolated to one subproject when possible: `jepa/` or `lejepa/`.
- Commit messages should be imperative and behavior-specific, e.g. `fix JEPA sequence shape handling`.
- PRs must include: scope, rationale, test evidence, and screenshots only for docs/figures/eval output changes.

## Build, Test, and Development Commands
- Prefer direct Python tooling; no top-level task runner is defined yet.
- Useful commands:
  - `cd jepa && python -m unittest tests.test_model.TestJEPA`
  - `cd jepa && black . && isort . && flake8 && mypy jepa`
  - `cd lejepa && pytest tests`
- For model/trainer work, always run at least one tiny synthetic smoke test in addition to unit tests.

## Coding Style and Conventions
- Use 4-space indentation.
- `snake_case` for functions, modules, and variables; `PascalCase` for classes.
- Keep modules focused and tensor contracts explicit.
- Add docstrings or comments when shape assumptions are non-obvious.
- Keep edits surgical; avoid broad refactors unless the task explicitly calls for architectural cleanup.

## Git Safety
- Safe defaults: `git status`, `git diff`, `git log`.
- No destructive git operations unless explicitly requested.
- No `git commit --amend` unless explicitly requested.
- Do not use `git add .`; use explicit paths or `git add -A`.

## Dependency Policy
- Before adding a dependency, do a quick health check:
  - maintenance activity
  - adoption and reputation
  - necessity for this workspace
- Prefer optional or isolated dependencies for docs, evaluation, and visualization tooling.

## Engineering Quality Bar
- Keep quality high for every change.
- Favor explicit architecture over convenience wrappers.
- For JEPA code, correctness of representation shapes, task construction, and loss contracts matters more than framework polish.
