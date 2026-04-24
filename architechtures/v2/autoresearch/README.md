# autoresearch

This directory is a port of Karpathy-style autoresearch onto the Intertwined H-JEPA codebase in the parent repo.

The key design decision is that `autoresearch/train.py` is **not** a second trainer. It is a thin wrapper around the real training code in `scripts/train_intertwined_hjepa.py` in the parent repo. The wrapper validates inputs, launches training, loads the final checkpoint, computes `val_bpb`, and appends a rich row to `results.tsv`.

## Files

```text
train.py         Wrapper entrypoint for autoresearch experiments
program.md       Agent operating instructions for this repo
results.tsv      Auto-initialized TSV ledger (ignored by git)
analysis.ipynb   Optional notebook for post-run analysis
prepare.py       Legacy upstream file; not used by this repo port
```

## Requirements

- Use the parent repo environment, not a duplicate trainer stack.
- Run `uv sync` at the repo root, or `cd autoresearch && uv sync` if you specifically want the local env.
- This workflow expects access to a Parameter Golf checkout so it can resolve:
  - cached FineWeb token shards
  - the matching SentencePiece tokenizer model

## Quick Start

Validate the setup without launching training:

```bash
uv run -- python autoresearch/train.py \
  --validate-only \
  --parameter-golf-root /path/to/parameter-golf
```

Run a tiny smoke experiment:

```bash
uv run -- python autoresearch/train.py \
  --profile smoke \
  --parameter-golf-root /path/to/parameter-golf \
  --description "baseline smoke" \
  --status baseline
```

Run a fuller baseline:

```bash
uv run -- python autoresearch/train.py \
  --profile full \
  --parameter-golf-root /path/to/parameter-golf \
  --description "baseline" \
  --status baseline
```

Run a compact JEPA-layer selection probe:

```bash
uv run -- python autoresearch/train.py \
  --profile full \
  --config sweep_configs/compact_16m.yaml \
  --parameter-golf-root /path/to/parameter-golf \
  --jepa-dropout-rate 0.1 \
  --auxiliary-layer-start 3 \
  --auxiliary-layer-stride 2 \
  --description "compact JEPA on blocks 4/6/8"
```

When any of those JEPA override flags are present, autoresearch writes the resolved YAML to the run directory as `autoresearch_config.yaml` and passes that file to the real trainer.

## What The Wrapper Does

1. Resolves the real dataset and tokenizer assets from Parameter Golf.
2. Calls the real JEPA trainer in `scripts/train_intertwined_hjepa`.
3. Loads the final trained model, usually from `latest.pt` and for smoke runs from the in-memory `--return-model` path.
5. Computes:
   - `val_bpb` as the primary experiment metric
   - eval LM / JEPA / SIGReg losses as supporting diagnostics
6. Appends a rich row to `results.tsv`.

## Results Ledger

`results.tsv` is tab-separated and intentionally more detailed than the upstream five-column file. The wrapper creates it automatically if missing.

Current columns:

```text
timestamp_utc
commit
run_name
profile
status
description
val_bpb
eval_loss
eval_loss_lm
eval_loss_jepa
eval_loss_sigreg
train_loss
train_loss_lm
train_loss_jepa
train_loss_sigreg
lambda_jepa
beta_sigreg
jepa_aux_dropped
jepa_dropout_fraction
peak_memory_gb
tokens_per_sec
tokens_processed
num_steps
wall_seconds
device
dtype
optimizer
lr
weight_decay
batch_size
seq_len
train_shards
eval_batches
config
checkpoint
run_dir
```

`val_bpb` is the keep/discard metric. The rest are there to explain why BPB moved.

## Workflow Rules

- Keep the wrapper thin. Do not fork the trainer into `autoresearch/train.py`.
- The real model/trainer lives in the parent repo.
- Never edit `jepa/` or `lejepa/`.
- Always do a smoke validation before a long run after structural changes.
- Prefer simpler improvements when BPB gains are marginal.

## Logging

For unattended runs, redirect output to a log file:

```bash
uv run -- python autoresearch/train.py ... > autoresearch/run.log 2>&1
```

At the end of a successful run the wrapper prints a stable summary block beginning with `---`, including `val_bpb`, eval losses, memory, run directory, and results path.
