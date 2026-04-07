# jepa-code

This repository is for studying and building JEPA-style architectures in a cleaner local workspace.

It currently contains three tracks:

- `jepa/`
  Reference-only legacy JEPA package. Do not edit.
- `lejepa/`
  Reference-only LeJEPA implementation. Do not edit.
- `src/text_jepa/`
  Active implementation area for new JEPA experiments in this repo.

## Current Focus

There are two active model paths under `src/text_jepa/`:

1. `Layer`
   A text-first masked latent-prediction JEPA with:
   - span masking
   - context encoder
   - target encoder
   - predictor
   - latent loss
   - EMA target updates

2. `LLM-JEPA`
   A paired-view causal language model path with:
   - standard LM loss
   - paired source/target views
   - predictor tokens on the source view
   - JEPA loss between source and target hidden states

The LLM-JEPA path is currently the most runnable local training path.

## Layout

```text
docs/                  Architecture notes and local design docs
scripts/               Training and utility scripts
src/text_jepa/         Active implementation code
tests/                 Unit tests and small integration tests
text-jepa-default.yaml Default tokenizer/model config
justfile               Common developer commands
```

## Important Boundaries

- Do not modify `jepa/`
- Do not modify `lejepa/`
- Durable project notes live in `.codex/learning.md`

## Environment

This repo expects a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers pyyaml pytest wandb datasets huggingface_hub
```

Depending on which path you use, you may also need additional packages already present in the local environment.

## Common Commands

Run focused tests:

```bash
just test-llm-jepa
just test-fineweb-dataloader
just test-train-checkpointing
```

Run all current top-level tests:

```bash
just all-tests
```

Download a tiny FineWeb sample:

```bash
just download-fineweb-sample
```

Run the Layer trainer:

```bash
just train-layer
```

Run the LLM-JEPA trainer:

```bash
just train-llm-jepa
```

For the smallest local LLM-JEPA sanity run:

```bash
source .venv/bin/activate
python scripts/train_llm_jepa.py \
  --steps 1 \
  --batch-size 1 \
  --max-length 128 \
  --save-every 0 \
  --train-file llm-jepa/datasets/synth_train.jsonl \
  --eval-file llm-jepa/datasets/synth_test.jsonl
```

## Notes on Local Runs

- On Apple Silicon, the trainers should prefer `mps` over `cpu`.
- The default LLM-JEPA local recipe is intentionally small because full Qwen-based training is expensive on a laptop.
- W&B defaults to offline mode unless online auth is available.

## Documentation

Layer-specific design docs:

- [`docs/text-jepa-plan.md`](/Users/namanchetwani/Projects/jepa-code/docs/text-jepa-plan.md)
- [`docs/text-jepa-flow.md`](/Users/namanchetwani/Projects/jepa-code/docs/text-jepa-flow.md)

Local durable learnings:

- [`.codex/learning.md`](/Users/namanchetwani/Projects/jepa-code/.codex/learning.md)
