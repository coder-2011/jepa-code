# Base JEPA Block Pseudocode

## Scope

This file describes the current base Intertwined H-JEPA block implemented in `intertwined_hjepa.py`.

The block is part of an autoregressive language model. Its predictor is not a disposable training head: the predicted compressed delta is projected back into the residual stream during training and inference.

Notation:

```text
B: batch size
L: sequence length
D: residual width
K: compressed JEPA width
```

## Student Block Contract

Each JEPA block receives a residual stream tensor:

```text
h_l: (B, L, D)
```

The block computes:

```text
h_l_normed     = RMSNorm(h_l)
h_l_post_attn  = h_l + CausalAttention_l(h_l_normed)
z_l            = CE_l(h_l_post_attn)
delta_l        = Pred_l(z_l)
update_l       = Proj_l(z_l + delta_l)
h_{l+1}        = h_l_post_attn + update_l
```

where:

```text
CE_l(h) = Compressor_l(RMSNorm_l(h))
z_l:       (B, L, K)
delta_l:   (B, L, K)
update_l:  (B, L, D)
```

The residual stream receives `Proj_l(z_l + delta_l)`, not `Proj_l(delta_l)`.

## Block Modules

Current JEPA block pieces:

```text
attn_norm:  RMSNorm(D)
attn:       causal MultiheadAttention(D), bias=False
ce_norm:    RMSNorm(D)
compressor: Linear(D, K) -> GELU -> Dropout -> Linear(K, K)
predictor:  RMSNorm(K) -> Linear(K, H) -> GELU -> Dropout -> Linear(H, K)
projector:  RMSNorm(K) -> Linear(K, D)
```

The compressor output `z_l` is the JEPA embedding for that layer.

## Full Model Shape

For config `depth = N`, the model uses:

```text
N - 1 JEPA blocks
1 final normal residual block
```

The final block is causal attention plus an MLP. It does not produce `z_l` or `delta_l`.

This gives every JEPA block a future target:

```text
JEPA block 0 targets JEPA block 1
...
top JEPA block targets the frozen output target encoder applied after the final block
```

## Teacher Targets

For non-top JEPA block `l`:

```text
target_z_l = stopgrad(CEbar_{l+1}(h_{l+1}_post_attn))
```

`CEbar_{l+1}` is the EMA copy of the next block's full CE path:

```text
EMA CE path = ema_ce_norm + ema_compressor
```

For the top JEPA block, there is no next JEPA block. Its target is:

```text
target_z_top = stopgrad(output_target_compressor(output_target_norm(h_final_post_attn)))
```

The output target modules are frozen after initialization.

## JEPA Loss

Current JEPA loss:

```text
L_jepa_l = MSE(delta_l, stopgrad(target_z_l) - z_l)
```

Only the teacher target is stopped. The student `z_l` is live, so JEPA gradients train:

```text
predictor_l
compressor_l / CE_l
ce_norm_l
post-attention path
earlier student computation
```

The EMA target path remains no-grad.

Masked loss uses the same formula but selects valid `(B, L)` positions before averaging over compressed width `K`.

## SIGReg Placement

SIGReg is applied directly to the cached encoder output embedding:

```text
SIGReg(z_l)
```

In code this is the already-computed `compressed[layer_index]`.

SIGReg is not applied to:

```text
delta_l
z_l + delta_l
projector(z_l + delta_l)
h_l_post_attn
the residual stream directly
```

Because `z_l` is used directly, SIGReg gradients flow through the full encoder path that produced the embedding, matching the LeJEPA interpretation of applying SIGReg to `f_theta(x)`.

## EMA Update

EMA tracks the CE path only:

```text
ema_ce_norm_l <- ema_momentum * ema_ce_norm_l + (1 - ema_momentum) * ce_norm_l
ema_compressor_l <- ema_momentum * ema_compressor_l + (1 - ema_momentum) * compressor_l
```

EMA excludes:

```text
attention
predictor
projector
embeddings
LM head
final block
output target modules
```

The EMA update runs after `optimizer.step()`.

## Initialization

The model uses explicit small initialization:

```text
Linear weights:    Normal(0, 0.02)
Linear biases:     0
Embedding weights: Normal(0, 0.02)
RMSNorm weights:   1
MHA in_proj_weight: Normal(0, 0.02)
```

EMA copies are created after student initialization so they start from the initialized CE path.

## Autoregressive Use

Generation is simple and uncached:

```text
1. Encode the current prefix.
2. Run the full model.
3. Read logits at the final position.
4. Sample or choose one next token.
5. Append it.
6. Repeat.
```

No KV cache is implemented yet.
