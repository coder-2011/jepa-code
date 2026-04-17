# Intertwined H-JEPA v1

This folder contains a small autoregressive language-model experiment built around JEPA-style local prediction. The model keeps the JEPA predictor active at inference time: each JEPA block predicts a compressed update, projects it back into the residual stream, and the final residual state feeds a normal next-token LM head.

The code is intentionally compact. The main implementation lives in `intertwined_hjepa.py`; training, checkpoint inspection, data helpers, and tests are separate.

## Files

```text
intertwined_hjepa.py          Model, losses, EMA target update, diagnostics
intertwined_hjepa.yaml        Default architecture and loss config
sigreg.py                    Sliced Epps-Pulley SIGReg implementation
text_helpers.py              HF tokenizer wrapper, token embeddings, LM head
data/dataset_helpers.py      Cached FineWeb shard loading and dataloaders
scripts/train_intertwined_hjepa.py
                              Training loop with W&B support and checkpoints
scripts/inspect_checkpoint.py
                              Eval, layer diagnostics, and simple generation
test/                        Model, loss, trainer, and checkpoint tests
test_dataset_helpers.py      Dataset helper tests
```

Design notes are in:

```text
base-jepa-block-pseudocode.md
intertwined-hjepa-plan.md
intertwined-hjepa-literature.md
learning.md
```

## Model

Notation:

```text
B: batch size
L: sequence length
D: residual width
K: compressed JEPA width
```

For `depth = N`, the model has:

```text
N - 1 JEPA blocks
1 final residual block
```

Each JEPA block computes:

```text
h_l_post_attn = h_l + CausalAttention_l(RMSNorm(h_l))
z_l           = Compressor_l(RMSNorm(h_l_post_attn))
delta_l       = Predictor_l(z_l)
h_{l+1}       = h_l_post_attn + Projector_l(z_l + delta_l)
```

Shapes:

```text
h_l:           (B, L, D)
h_l_post_attn: (B, L, D)
z_l:           (B, L, K)
delta_l:       (B, L, K)
logits:        (B, L, vocab_size)
```

The final block is a normal causal residual block. It does not produce `z_l` or `delta_l`. The final logits are:

```text
logits = LMHead(final_norm(h_final))
```

The LM head can tie its weights to the token embedding.

## Targets And Losses

The training objective is:

```text
loss = loss_lm
     + lambda_jepa_eff * loss_jepa
     + beta_sigreg_eff * loss_sigreg
```

`lambda_jepa_eff` and `beta_sigreg_eff` use linear warmup schedules controlled by the YAML config.

### LM Loss

The LM loss is standard next-token cross entropy:

```text
loss_lm = CE(logits[:, :-1], labels[:, 1:])
```

### JEPA Loss

For non-top JEPA block `l`, the target is the EMA copy of the next block's CE path:

```text
target_z_l = stopgrad(EMA_CE_{l+1}(h_{l+1}_post_attn))
```

For the top JEPA block, the target comes from a frozen output target encoder after the final block:

```text
target_z_top = stopgrad(output_target_compressor(output_target_norm(h_final_post_attn)))
```

The JEPA loss is:

```text
loss_jepa_l = MSE(delta_l, stopgrad(target_z_l) - z_l)
```

Only the teacher target is stopped. The student `z_l` is live, so JEPA trains the predictor and the student CE path.

### SIGReg Loss

SIGReg is applied directly to the cached encoder embedding:

```text
loss_sigreg_l = SIGReg(z_l)
```

It is not applied to `delta_l`, `z_l + delta_l`, the projected update, or the residual stream directly. Because each `z_l` is the live tensor from the forward pass, SIGReg gradients flow through the student encoder path that produced it.

## EMA

EMA tracks only the CE path for each JEPA block:

```text
ema_ce_norm_l
ema_compressor_l
```

EMA does not track attention, predictor, projector, embeddings, LM head, final block, or the frozen output target encoder.

Call order during training:

```text
loss.backward()
optimizer.step()
model.update_ema(step_index)
model.zero_grad(set_to_none=True)
```

## Default Config

The default config is `intertwined_hjepa.yaml`:

```yaml
vocab_size: 1024
max_length: 128
residual_dim: 256
compressed_dim: 128
depth: 4
num_heads: 4
predictor_hidden_dim: 256
dropout: 0.0
ema_momentum: 0.996
ema_momentum_final: 0.996
ema_warmup_steps: 0
lambda_jepa: 1.0
jepa_warmup_steps: 100
beta_sigreg: 0.04
sigreg_warmup_steps: 100
sigreg_num_slices: 256
sigreg_t_max: 3.0
sigreg_n_points: 17
tie_weights: true
```

`beta_sigreg` is the coefficient on SIGReg. It corresponds to the LeJEPA-style lambda in `L_pred + lambda * L_sig`.

## Setup

Use `uv` for environment management in this repo. Install dependencies and create the local environment with:

```bash
uv sync
```

Run project commands with `uv run` so they execute inside the synced environment.

Use `uv lock` when you need to refresh the lockfile, and avoid manual `venv`/`pip` setup for the normal workflow.

W&B is included in the project dependencies, so no separate install step is needed.
This repo now targets a Blackwell-friendly stack: PyTorch `2.9.1` with CUDA `13.0` wheels, `torchao>=0.15.0`, and `flash-attn-4==4.0.0b9` cu13 for attention.

The training script expects cached Parameter Golf FineWeb shards unless `--dataset-root` points directly at a compatible dataset folder.

## Tests

Run all tests:

```bash
uv run -- pytest test test_dataset_helpers.py
```

Model-focused tests:

```bash
uv run -- pytest test/test_intertwined_hjepa_shapes.py test/test_intertwined_hjepa_training_step.py
```

Trainer tests:

```bash
uv run -- pytest test/test_train_intertwined_hjepa.py
```

## Training

Small local smoke run:

```bash
uv run -- python -m scripts.train_intertwined_hjepa \
  --parameter-golf-root /Users/namanchetwani/Projects/parameter-golf \
  --variant sp1024 \
  --device mps \
  --batch-size 2 \
  --seq-len 64 \
  --train-shards 1 \
  --max-steps 300 \
  --log-every 25 \
  --eval-every 100 \
  --eval-batches 2 \
  --save-every 0 \
  --run-name sanity-overfit \
  --wandb-mode disabled
```

Checkpointing:

```bash
uv run -- python -m scripts.train_intertwined_hjepa \
  --parameter-golf-root /Users/namanchetwani/Projects/parameter-golf \
  --variant sp1024 \
  --device mps \
  --batch-size 2 \
  --seq-len 64 \
  --train-shards 1 \
  --max-steps 2000 \
  --log-every 50 \
  --eval-every 500 \
  --eval-batches 2 \
  --save-every 1000 \
  --run-name current-2000-save1000 \
  --wandb-mode disabled
```

Checkpoints are written to:

```text
runs/intertwined_hjepa/<run-name>/
```

## Inspecting A Checkpoint

```bash
./.venv/bin/python -m scripts.inspect_checkpoint \
  --checkpoint runs/intertwined_hjepa/current-2000-save1000/latest.pt \
  --parameter-golf-root /Users/namanchetwani/Projects/parameter-golf \
  --variant sp1024 \
  --device mps \
  --batch-size 2 \
  --seq-len 64 \
  --eval-batches 2 \
  --max-new-tokens 40
```

The inspector reports:

```text
eval loss
LM / JEPA / SIGReg loss
per-layer z_std_mean and z_std_min
effective rank
delta norm and target-delta norm
sample generations
warnings for suspicious collapse or delta scale
```

## Diagnostics To Watch

Healthy smoke-test signs:

```text
train/loss_lm decreases
eval/loss_lm decreases or at least stays sane
loss_jepa remains finite
loss_sigreg remains finite
grad_norm does not explode
z_std_min_layer_i does not go to 0
```

`z_std_min_layer_i` is a collapse signal. It is computed by flattening batch and sequence for a layer's `z_l`:

```text
z_l: (B, L, K) -> (B * L, K)
std over B * L for each compressed dimension
min over K
```

If `z_std_min_layer_i` goes near zero, at least one compressed feature is nearly constant across the sampled token positions.

## W&B

W&B is disabled by default:

```bash
--wandb-mode disabled
```

Offline logging:

```bash
--wandb-mode offline
```

Online logging:

```bash
--wandb-mode online --wandb-project intertwined-hjepa
```

The trainer logs total losses, per-layer JEPA/SIGReg losses, `z` statistics, delta norms, gradient norm, step time, and tokens/sec.

## Current Limitations

Generation is uncached. There is no KV cache yet.

The model is still an experiment. Current local tests check shape contracts, gradient routing, checkpoint loading, trainer behavior, and dataset helpers; they do not prove scaling behavior.

The current SIGReg setup is layerwise: it applies to every JEPA block's `z_l`. That is stronger than a strict final-encoder-only reading of LeJEPA, but it matches the current architecture's explicit local representation contract.
