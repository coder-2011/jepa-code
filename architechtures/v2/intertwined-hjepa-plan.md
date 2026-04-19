# Intertwined H-JEPA Plan

## Purpose

Intertwined H-JEPA is a predictive residual-stream language model. Each JEPA block compresses its post-attention residual state, predicts a compressed delta toward a future target representation, and injects the enriched compressed state back into the dense residual stream.

The predictor remains active at inference. This is not a standard JEPA setup where the predictor can be discarded after training.

## Current Architecture

For config `depth = N`, the implemented model contains:

```text
N - 1 JEPA blocks
1 final residual block
```

The final residual block is causal attention plus an MLP and produces the final language-model residual state.

For JEPA block `l`:

```text
h_l_normed     = RMSNorm(h_l)
h_l_post_attn  = h_l + CausalAttention_l(h_l_normed)
z_l            = CE_l(h_l_post_attn)
delta_l        = Pred_l(z_l)
h_{l+1}        = h_l_post_attn + Proj_l(z_l + delta_l)
```

where:

```text
CE_l(h) = Compressor_l(RMSNorm_l(h))
z_l:     (B, L, K)
delta_l: (B, L, K)
h_l:     (B, L, D)
```

The final logits are:

```text
logits = LMHead(final_norm(h_final))
```

The LM head may tie weights with the token embedding.

## Targets

Every JEPA block has a target.

For every JEPA block, the target is the same-depth EMA CE path at the next token position:

```text
target_z_l[:, t] = stopgrad(EMA_CE_l(h_l_post_attn)[:, t+1])
```

`EMA_CE_l` is the EMA copy of the current block's CE path:

```text
ema_ce_norm + ema_compressor
```

## Losses

The training objective is:

```text
loss = loss_lm
     + lambda_jepa_eff * loss_jepa
     + beta_sigreg_eff * loss_sigreg
```

Warmups are scalar linear warmups over `jepa_warmup_steps` and `sigreg_warmup_steps`.

### LM Loss

The LM loss is normal next-token cross entropy. The loader already shifts the
targets, so the loss consumes aligned `(logits[:, t], labels[:, t])` pairs:

```text
loss_lm = CE(logits, labels)
```

### JEPA Loss

Current JEPA loss:

```text
loss_jepa_l = MSE(delta_l, stopgrad(target_z_l) - z_l)
```

The target is stopped. The student `z_l` is not stopped. Therefore JEPA trains both the predictor and the encoder path.

This replaced the older predictor-only form:

```text
MSE(delta_l, stopgrad(target_z_l - z_l))
```

which detached `z_l` and did not train the encoder path through JEPA.

### SIGReg Loss

SIGReg is applied to the actual cached encoder embedding:

```text
loss_sigreg_l = SIGReg(z_l)
```

`z_l` is the direct analog of LeJEPA's `f_theta(x)` output embedding.

SIGReg is not applied to:

```text
delta_l
z_l + delta_l
projector(z_l + delta_l)
the dense residual stream
```

Because SIGReg uses the cached `z_l`, gradients flow through the full encoder path that produced `z_l`.

## EMA Contract

EMA tracks the teacher CE path:

```text
ema_ce_norm_l
ema_compressor_l
```

EMA excludes:

```text
attention
predictor
projector
embeddings
LM head
final residual block
```

The EMA path is never optimized by AdamW and only changes in `update_ema()` after `optimizer.step()`.

Current EMA momentum in YAML:

```text
ema_momentum: 0.996
```

## Initialization

The model uses explicit small initialization:

```text
Linear weights:    Normal(0, 0.02)
Linear biases:     0
Embedding weights: Normal(0, 0.02)
RMSNorm weights:   1
MHA in_proj_weight: Normal(0, 0.02)
```

This replaced PyTorch default embedding initialization, which made initial logits too hot.

## Current YAML Defaults

Current important knobs:

```text
lambda_jepa: 1.0
jepa_warmup_steps: 100
beta_sigreg: 0.04
sigreg_warmup_steps: 100
ema_momentum: 0.996
dropout: 0.0
```

The LeJEPA-style `lambda` in `L_pred + lambda * L_sig` corresponds to this codebase's `beta_sigreg`.

## Diagnostics

The model reports:

```text
loss_lm
loss_jepa
loss_sigreg
loss_jepa_layer_i
loss_sigreg_layer_i
z_variance_layer_i
z_std_mean_layer_i
z_std_min_layer_i
delta_norm_layer_i
```

`z_std_min` is a worst-dimension collapse check after flattening batch and sequence:

```text
z_l:      (B, L, K)
z_flat:   (B * L, K)
z_std:    std(z_flat, dim=0)
z_std_min = min(z_std)
```

## Current Empirical Notes

On local M2 runs with `B=2`, `L=64`, `D=256`, `K=128`:

```text
small init made initial LM loss sane: around log(1024)
JEPA must flow into z_l / CE for healthy representations
SIGReg must apply directly to z_l with full encoder gradients
beta_sigreg=0.04 works after the SIGReg path fix
```

A 2000-step run with corrected SIGReg and `beta_sigreg=0.04` produced healthy non-final JEPA block stats:

```text
z_std_mean_layer_0..2 around 1.0
z_std_min_layer_0..2 roughly 0.4-0.5 late in training
eval/loss_lm around 5.96
eval/loss_sigreg around 16
```

The model is still far from coherent language generation at 256k tokens; this is a training-loop and representation-health checkpoint, not a finished language model.

## Training And Inference

Training uses cached Parameter-Golf FineWeb shards through `scripts.train_intertwined_hjepa`.

Typical local smoke command:

```bash
./.venv/bin/python -m scripts.train_intertwined_hjepa \
  --parameter-golf-root /Users/namanchetwani/Projects/parameter-golf \
  --variant sp1024 \
  --device mps \
  --batch-size 2 \
  --seq-len 64 \
  --train-shards 1 \
  --max-steps 2000 \
  --log-every 250 \
  --eval-every 500 \
  --eval-batches 2 \
  --save-every 2000 \
  --run-name smoke \
  --wandb-mode disabled
```

Inference/inspection uses:

```bash
./.venv/bin/python -m scripts.inspect_checkpoint \
  --checkpoint runs/intertwined_hjepa/<run>/latest.pt \
  --parameter-golf-root /Users/namanchetwani/Projects/parameter-golf \
  --variant sp1024 \
  --device mps
```

Generation currently reruns the full context each token. There is no KV cache.
