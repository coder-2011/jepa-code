# Base JEPA Block Pseudocode

## Scope

This file sketches only the base Intertwined H-JEPA block.

Tokenizer and embeddings are outside this block. They can be borrowed from existing repo code. The base JEPA block itself should be simple:

```text
compress current state -> predict delta -> project enriched state -> add to residual stream
```

The first pass contains only the JEPA transform.

## Current Gimmicks

These are the deliberate mechanisms currently in the design:

1. **Compressed JEPA space**
   `CE_L` maps residual width `D` into compressed width `K`.

2. **Delta prediction**
   `Pred_L` predicts `delta_L`, not the full target representation.

3. **Enriched residual update**
   The residual update uses `Proj_L(z_L + delta_L)`, not just `Proj_L(delta_L)`.

4. **One-layer-future EMA compressor target**
   The target for layer `L` comes from the EMA compressor copy of layer `L + 1`.

5. **Stop-gradient delta target**
   The JEPA loss compares `delta_L` to `target_z_L - sg(z_L)`.

6. **LM head outside the block**
   The final model still has an LM head, but the base JEPA block does not know about vocabulary logits.

Out of scope for the first pass: FFN residual sublayers and custom tokenizer work.

## Contract

For layer `L` inside one generation/training forward:

```text
input:
  x_L: residual stream entering this JEPA block
       generation shape: (B, 1, D) or (B, D)
       teacher-forced training shape: (B, T, D)

student forward:
  z_L      = CE_L(x_L)
  delta_L  = Pred_L(z_L)
  update_L = Proj_L(z_L + delta_L)
  x_{L+1}  = x_L + update_L

stored for later:
  x_L
  z_L
  delta_L

target/loss later, after full forward:
  target_z_L   = sg(CEbar_{L+1}(x_{L+1}))
  target_delta = target_z_L - sg(z_L)
  L_jepa_L     = MSE(delta_L, target_delta)
```

Autoregressive usage:

```text
1. Embed the current generated prefix or current token state.
2. Run the JEPA stack once.
3. LM head emits logits for the next token.
4. Sample/choose one token.
5. Append that token to the generated sequence.
6. Run the model again for the next token.
```

The first pass does not implement caching. It can simply rerun the full prefix each generation step if using prefix-shaped tensors.

## Error Handling Policy

Keep explicit checks minimal. Let PyTorch fail naturally for ordinary shape or dtype mistakes.

Only add manual checks for invariants that would otherwise fail later with a confusing error:

```text
depth must be at least 2
compressed dims must match between delta, z, and target
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
        dropout: float = 0.0,
    ):
        super().__init__()

        # CE_L: compress residual stream into JEPA space.
        self.compressor = Sequential(
            RMSNorm(residual_dim),
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
def forward_student(self, x_l):
    # x_l: (B, T, D), or (B, D) for a single-token generation step

    z_l = self.compressor(x_l)
    delta_l = self.predictor(z_l)
    update_l = self.projector(z_l + delta_l)
    x_next = x_l + update_l

    return {
        "x_next": x_next,
        "x": x_l,
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

Do not EMA the predictor or projector in the first pass. The EMA branch is only a target-value branch.

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

Prefer this over a manual parameter loop:

```python
for p in self.ema_compressors.parameters():
    p.requires_grad_(False)
```

## Teacher Target Forward

The teacher target for block `L` is computed by the EMA compressor copy of the **next** JEPA block, using the next layer's student state.

```python
@torch.no_grad()
def compute_jepa_target_for_layer(
    layer_index,
    ema_compressors,
    stored_x_next,
    target_projectors,
):
    # For L, use EMA compressor L+1 and x_{L+1}.
    next_ema_compressor = ema_compressors[layer_index + 1]
    x_next = stored_x_next[layer_index]

    target_z_l = next_ema_compressor(x_next)

    # If compressed dims differ by layer, project to this layer's K.
    # For v1 with one K everywhere, this can be Identity.
    target_z_l = target_projectors[layer_index](target_z_l)
    return target_z_l.detach()
```

Inside each block:

```python
def forward_target_compression(self, x):
    return self.compressor(x)
```

Important implementation choice:

```text
Every layer needs a compressor and EMA compressor. JEPA losses are computed for
layers 0 through depth-2, because layer depth-1 has no future layer target.
```

## Loss For One Block

```python
def jepa_delta_loss(delta_l, z_l, target_z_l, valid_mask=None):
    # delta_l: (B, T, K)
    # z_l:     (B, T, K)
    # target_z_l: (B, T, K)

    target_delta = target_z_l.detach() - z_l.detach()
    error = (delta_l - target_delta).pow(2)

    if valid_mask is not None:
        # valid_mask: (B, T), True where loss is active.
        mask = valid_mask.unsqueeze(-1).to(error.dtype)
        # Explicit check is useful here; silently dividing by zero hides bad batches.
        if mask.sum() == 0:
            raise ValueError("valid_mask selects no JEPA loss positions")
        denom = mask.sum().mul(error.shape[-1])
        return (error * mask).sum() / denom

    return error.mean()
```

## Full Model Loop Using The Block

This is a stack of JEPA blocks only.

```python
def forward(input_ids, labels=None, valid_mask=None, step=None):
    x = borrowed_embeddings(input_ids)

    stored = {
        "x_in": [],
        "x_next": [],
        "z": [],
        "delta": [],
    }

    # Student forward first.
    for layer_index, block in enumerate(student_blocks):
        out = block.forward_student(x)
        x = out["x_next"]

        stored["x_in"].append(out["x"])
        stored["x_next"].append(out["x_next"])
        stored["z"].append(out["z"])
        stored["delta"].append(out["delta"])

    logits = lm_head(final_norm(x))
    loss_main = next_token_loss(logits, labels, valid_mask)

    # Targets and local JEPA losses second.
    jepa_losses = []
    for layer_index in range(num_layers - 1):
        target_z = compute_jepa_target_for_layer(
            layer_index=layer_index,
            ema_compressors=ema_compressors,
            stored_x_next=stored["x_next"],
            target_projectors=target_projectors,
        )

        jepa_losses.append(
            jepa_delta_loss(
                delta_l=stored["delta"][layer_index],
                z_l=stored["z"][layer_index],
                target_z_l=target_z,
                valid_mask=valid_mask,
            )
        )

    lambda_eff = warmup(config.lambda_jepa, step, config.jepa_warmup_steps)

    loss_jepa = weighted_sum(jepa_losses)
    loss = loss_main + lambda_eff * loss_jepa

    return {
        "loss": loss,
        "loss_main": loss_main,
        "loss_jepa": loss_jepa,
        "logits": logits,
        "diagnostics": build_diagnostics(stored, jepa_losses),
    }
```

LM-head weight tying, if compatible:

```python
if tie_weights:
    lm_head.weight = token_embedding.weight
```

Only do this when both weights are shaped `(vocab_size, residual_dim)`. If the borrowed embedding path does not expose a compatible token embedding table, keep the LM head untied.

Initial config:

```text
lambda_jepa = 0.1
jepa_warmup_steps = small nonzero warmup for real training, 0 for shape tests
```

## Training Step Pseudocode

```python
optimizer.zero_grad(set_to_none=True)

out = model(
    input_ids=batch["input_ids"],
    labels=batch["labels"],
    valid_mask=batch.get("valid_mask"),
    step=global_step,
)

out["loss"].backward()
clip_grad_norm_(model.student_parameters(), max_norm)
optimizer.step()

# EMA compressor update always happens after optimizer.step().
model.update_ema()
```

EMA update helper:

```python
@torch.no_grad()
def update_ema(self):
    for ema_param, param in zip(
        self.ema_compressors.parameters(),
        self.compressors.parameters(),
    ):
        ema_param.mul_(self.ema_momentum).add_(
            param,
            alpha=1.0 - self.ema_momentum,
        )
```

## First Tests

```text
test_block_student_forward_shapes
  x_next: (B, T, D)
  z:      (B, T, K)
  delta:  (B, T, K)

test_jepa_delta_loss_detaches_target
  target_z and z do not receive gradients through target_delta
  delta receives gradients

test_ema_target_no_grad
  target computation creates no graph

test_ema_updates_only_after_optimizer
  EMA compressor parameters change after model.update_ema()
  EMA compressor parameters do not have optimizer gradients
```
