# Base JEPA Block Pseudocode

## Scope

This file describes the current base Intertwined H-JEPA block implemented in `intertwined_hjepa.py`.

The block is part of an autoregressive language model. Its JEPA predictor is an auxiliary training head: the predicted compressed delta is used by the JEPA loss, but it is not projected back into the residual stream.

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
update_l       = MLP_l(h_l_post_attn)
h_{l+1}        = h_l_post_attn + update_l
```

where:

```text
CE_l(h) = Compressor_l(RMSNorm_l(h))
z_l:       (B, L, K)
delta_l:   (B, L, K)
update_l:  (B, L, D)
```

The residual stream is updated by its own transition MLP. The auxiliary latent path (`z_l`, `delta_l`) is decoupled from the residual update.

## Block Modules

Current JEPA block pieces:

```text
attn_norm:  RMSNorm(D)
attn:       causal MultiheadAttention(D), bias=False
ce_norm:    RMSNorm(D)
compressor: Linear(D, K) -> GELU -> Dropout -> Linear(K, K)
predictor:  RMSNorm(K) -> Linear(K, H) -> GELU -> Dropout -> Linear(H, K)
transition: RMSNorm(D) -> Linear(D, H) -> GELU -> Dropout -> Linear(H, D)
```

The compressor output `z_l` is the JEPA embedding for that layer. The predictor output `delta_l` is trained to match the next-token delta in that same layer's EMA-compressed representation.

## Full Model Shape

For config `depth = N`, the model uses:

```text
N - 1 JEPA blocks
1 final normal residual block
```

The final block is causal attention plus an MLP. It does not produce `z_l` or `delta_l`.

This gives every JEPA block a same-layer next-token target:

```text
JEPA block l uses EMA_CE_l(h_l_post_attn) as its target sequence
delta_l[:, t] predicts EMA_CE_l(h_l_post_attn)[:, t+1] - z_l[:, t]
```

## Teacher Targets

For JEPA block `l`:

```text
target_z_l = stopgrad(EMA_CE_l(h_l_post_attn))
```

`EMA_CE_l` is the EMA copy of the same block's CE path:

```text
EMA CE path = ema_ce_norm + ema_compressor
```

The temporal shift happens in the loss:

```text
target_delta_l[:, t] = target_z_l[:, t+1] - z_l[:, t]
```

The last sequence position has no next-token JEPA target and is masked out.

## JEPA Loss

Current JEPA loss:

```text
L_jepa_l = MSE(delta_l[:, t], stopgrad(target_z_l[:, t+1]) - z_l[:, t])
```

In the vectorized implementation this is evaluated at positions `t = 0..L-2`.

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
embeddings
LM head
final block
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
