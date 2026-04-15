# Base JEPA Block Pseudocode

## Scope

This file sketches only the base Intertwined H-JEPA block.

Tokenizer and embeddings are outside this block. They can be borrowed from existing repo code. The base JEPA block itself should stay compact:

```text
pre-norm attention -> compress post-attention state -> predict delta -> project enriched state -> add to residual stream
```

The first pass contains:

- pre-norm self-attention
- compressor
- predictor
- projector

Out of scope for the first pass:

- FFN residual sublayer
- SIGReg / VICReg
- tokenizer implementation work
- caching

## Current Mechanics

These are the deliberate mechanisms currently in the design:

1. **Pre-norm attention**
   `Attention_L` runs on `LayerNorm(h_L)`, and its residual output defines the post-attention state.

2. **Compressed JEPA space**
   `CE_L` maps residual width `D` into compressed width `K`.

3. **Delta prediction**
   `Pred_L` predicts `delta_L`, not the full target representation.

4. **Enriched residual update**
   The residual update uses `Proj_L(z_L + delta_L)`, not just `Proj_L(delta_L)`.

5. **One-layer-future EMA compressor target**
   The target for layer `L` comes from the EMA compressor copy of layer `L + 1`, applied to the next layer's normalized post-attention state.

6. **Stop-gradient delta target**
   The JEPA loss compares `delta_L` to `target_z_L - sg(z_L)`.

7. **LM head outside the block**
   The final model still has an LM head, but the base JEPA block does not know about vocabulary logits.

## Contract

For layer `L` inside one generation/training forward:

```text
input:
  h_L: residual stream entering this JEPA block
       generation shape: (B, 1, D) or (B, D)
       teacher-forced training shape: (B, T, D)

student forward:
  h_L_normed     = LayerNorm(h_L)
  h_L_post_attn  = h_L + Attention_L(h_L_normed)
  z_L            = CE_L(LayerNorm(h_L_post_attn))
  delta_L        = Pred_L(z_L)
  update_L       = Proj_L(z_L + delta_L)
  h_{L+1}        = h_L_post_attn + update_L

stored for later:
  h_L
  h_L_post_attn
  z_L
  delta_L

target/loss later, after full forward:
  target_z_L   = sg(CEbar_{L+1}(LayerNorm(h_{L+1}_post_attn)))
  target_delta = target_z_L - sg(z_L)
  L_jepa_L     = MSE(delta_L, target_delta)
```

Autoregressive usage:

```text
1. Embed the current prefix.
2. Run the JEPA stack once.
3. LM head emits logits for the next token.
4. Sample or choose one token.
5. Append that token.
6. Run the model again.
```

The first pass does not implement caching. It can rerun the full prefix each generation step.

## Error Handling Policy

Keep explicit checks minimal. Let PyTorch fail naturally for ordinary shape or dtype mistakes.

Only add manual checks for invariants that would otherwise fail later with a confusing error:

```text
depth must be at least 2
delta, z, and target_z must have the same shape in JEPA loss
valid_mask must not select zero elements when used for a loss
```

## Minimal Class Shape

```python
class BaseJEPABlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,     # D
        compressed_dim: int,   # K
        predictor_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(residual_dim)
        self.attn = CausalSelfAttention(
            residual_dim=residual_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ce_norm = RMSNorm(residual_dim)

        # CE_L: compress normalized post-attention state into JEPA space.
        self.compressor = Sequential(
            Linear(residual_dim, compressed_dim),
            GELU(),
            Dropout(dropout),
            Linear(compressed_dim, compressed_dim),
        )

        # Pred_L: predict the delta inside compressed space.
        self.predictor = Sequential(
            RMSNorm(compressed_dim),
            Linear(compressed_dim, predictor_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(predictor_hidden_dim, compressed_dim),
        )

        # Proj_L: send enriched compressed representation back to residual stream.
        self.projector = Sequential(
            RMSNorm(compressed_dim),
            Linear(compressed_dim, residual_dim),
        )
```

## Student Forward

```python
def forward_student(self, h_l):
    # h_l: (B, T, D), or (B, D) for a single-token generation step

    h_l_normed = self.attn_norm(h_l)
    attn_out = self.attn(h_l_normed)
    h_l_post_attn = h_l + attn_out

    z_l = self.compressor(self.ce_norm(h_l_post_attn))
    delta_l = self.predictor(z_l)
    update_l = self.projector(z_l + delta_l)
    h_next = h_l_post_attn + update_l

    return {
        "h_next": h_next,
        "h": h_l,
        "post_attn": h_l_post_attn,
        "z": z_l,
        "delta": delta_l,
    }
```

## EMA Handling

EMA is only for compressors in the first pass.

```text
student compressor: CE_L
teacher compressor: CEbar_L
```

Initialization:

```text
CEbar_L <- CE_L
```

Post-optimizer update:

```text
CEbar_L <- ema_momentum * CEbar_L + (1 - ema_momentum) * CE_L
```

Rules:

```text
CEbar_L is never optimized by AdamW.
CEbar_L is only changed by hard-copy init or update_ema().
Targets are computed under torch.no_grad().
The EMA update runs only after optimizer.step().
```

Do not EMA the predictor or projector in the first pass.

Clean helper:

```python
def make_ema_copy(module: nn.Module) -> nn.Module:
    ema = copy.deepcopy(module)
    ema.requires_grad_(False)
    ema.eval()
    return ema
```

Usage:

```python
self.ema_compressors = make_ema_copy(self.compressors)
```

## Teacher Target Forward

The teacher target for block `L` is computed by the EMA compressor copy of the **next** JEPA block, using the next layer's normalized student post-attention state.

```python
@torch.no_grad()
def compute_jepa_target_for_layer(
    layer_index,
    ema_compressors,
    ce_norms,
    stored_post_attn,
):
    next_ema_compressor = ema_compressors[layer_index + 1]
    next_post_attn = stored_post_attn[layer_index + 1]

    target_z_l = next_ema_compressor(
        ce_norms[layer_index + 1](next_post_attn)
    )
    return target_z_l.detach()
```

Important implementation choice:

```text
Every layer needs a compressor and EMA compressor. JEPA losses are computed for
layers 0 through depth-2, because layer depth-1 has no future layer target.
```

## Loss For One Block

```python
def jepa_delta_loss(delta_l, z_l, target_z_l, valid_mask=None):
    if delta_l.shape != z_l.shape or delta_l.shape != target_z_l.shape:
        raise ValueError("delta_l, z_l, and target_z_l must have the same shape")

    target_delta = target_z_l.detach() - z_l.detach()
    error = (delta_l - target_delta).pow(2)

    if valid_mask is not None:
        mask = valid_mask.unsqueeze(-1).to(error.dtype)
        if mask.sum() == 0:
            raise ValueError("valid_mask selects no JEPA loss positions")
        denom = mask.sum().mul(error.shape[-1])
        return (error * mask).sum() / denom

    return error.mean()
```

For v1, use plain MSE only:

```text
L_jepa_L = MSE(delta_L, target_z_L.detach() - z_L.detach())
```

## Integration In A Full Model

The outer model is responsible for:

- token + position embeddings
- stacking JEPA blocks
- storing post-attention states
- LM head
- next-token CE loss
- EMA updates

The simplest full-model contract is:

```text
input_ids -> embeddings -> h_0
h_0 -> block_0 -> h_1
h_1 -> block_1 -> h_2
...
h_last -> final_norm -> lm_head -> logits
```

And later:

```text
for each layer L < depth - 1:
  target_z_L = EMA compressor at L+1 applied to next post-attention state
  loss_jepa_L = MSE(delta_L, target_z_L - sg(z_L))
```
