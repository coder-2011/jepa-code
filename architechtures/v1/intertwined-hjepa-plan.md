# Intertwined H-JEPA Plan

## Purpose

This document defines the first implementation plan for an **Intertwined Hierarchical JEPA** architecture.

The design goal is not to build a standard JEPA pretraining head. The first pass is a compact autoregressive residual-stream model in which each non-final layer learns a JEPA-style predictive delta toward a one-layer-future EMA teacher state, and that delta is added back into the residual stream during both training and inference.

## Evidence Base

Primary local references:

- `docs/papers/JEPA.pdf`: I-JEPA paper.
- `docs/papers/LeJEPA.pdf`: LeJEPA paper.
- `docs/text-jepa-plan.md`: current standard text-JEPA baseline in this repo.
- `docs/text-jepa-flow.md`: current standard text-JEPA tensor contracts.
- `docs/intertwined-hjepa-literature.md`: literature map and architecture implications.

External papers to keep in the design loop:

- I-JEPA: latent prediction from context blocks to target-block representations, with masking strategy central to semantic level.
  <https://arxiv.org/abs/2301.08243>
- BYOL: online network predicts an EMA target network representation, and the target is updated by slow moving average.
  <https://arxiv.org/abs/2006.07733>
- data2vec: masked-view student predicts contextualized latent representations from a teacher/self-distillation setup across modalities.
  <https://arxiv.org/abs/2202.03555>
- Masked Siamese Networks: masked-view representation matching to unmasked-view representation, useful precedent for sparse/masked computation tradeoffs.
  <https://arxiv.org/abs/2204.07141>
- V-JEPA: feature prediction as a standalone objective for video without reconstruction, negatives, or extra supervision.
  <https://arxiv.org/abs/2404.08471>
- VICReg: explicit variance and covariance regularization to prevent representational collapse.
  <https://arxiv.org/abs/2105.04906>
- LeJEPA: predictive JEPA objective plus SIGReg, arguing for isotropic Gaussian embeddings and a direct anti-collapse regularizer.
  <https://arxiv.org/abs/2511.08544>
- LLM-JEPA: applies JEPA-style embedding-space objectives to language models and motivates keeping an LM head in the design.
  <https://arxiv.org/abs/2509.14252>
- NextLat: adds next-latent prediction to next-token training and is a close auxiliary-loss baseline for our residual-delta path.
  <https://arxiv.org/abs/2511.05963>
- PredNet: predictive-coding network where layers make local predictions and pass deviations upward.
  <https://arxiv.org/abs/1605.08104>

## Core Claim

Intertwined H-JEPA should be treated as a **predictive residual-stream architecture**, not as a disposable auxiliary JEPA head.

In ordinary JEPA, the predictor is often a training-time module used to make the context representation match the target representation. In this architecture, the predictor persists at inference and its output is added into the residual stream. That makes the predictor part of the model's computation.

## Architecture Definition

For layer `l`, define:

- `x`: input token ids
- `h_0`: token + position embeddings
- `h_l`: residual stream entering JEPA block `l`
- `h_l_post_attn`: post-attention residual state for layer `l`
- `z_l`: compressed representation of `h_l_post_attn`
- `CE_l`: online compressor at layer `l`
- `CEbar_l`: EMA copy of compressor `CE_l`
- `P_l`: predictor at layer `l`
- `d_l`: predicted delta at layer `l`

The online path is:

```text
x -> token_embedding + position_embedding -> h_0
h_l_normed     = LayerNorm(h_l)
h_l_post_attn  = h_l + Attention_l(h_l_normed)
z_l            = CE_l(LayerNorm(h_l_post_attn))
d_l            = P_l(z_l)
h_{l+1}        = h_l_post_attn + Proj_l(z_l + d_l)
h_depth -> final_norm -> lm_head -> token logits
```

The teacher target for non-final layer `l` is:

```text
target_z_l = stopgrad(CEbar_{l+1}(LayerNorm(h_{l+1}_post_attn)))
```

The layer loss is:

```text
target_delta_l = target_z_l - stopgrad(z_l)
loss_jepa_l = MSE(d_l, target_delta_l)
```

The total predictive loss is:

```text
loss_jepa = sum_l alpha_l * loss_l, for l in [0, depth - 2]
```

The final layer has no natural `l + 1` teacher, so v1 should skip the final-layer JEPA loss.

## Autoregressive Generation Loop

The intended inference loop is autoregressive:

```text
1. Start with an input prefix.
2. Run the full JEPA stack.
3. Use the LM head to produce logits for one next token.
4. Pick or sample that token.
5. Append it to the sequence.
6. Run the model again.
```

So each generation pass produces one new token. Training can still use teacher forcing over a full sequence for efficiency:

```text
training logits:   (B, L, vocab_size)
generation logits: (B, vocab_size) for the final/current position
```

The first implementation does not need KV caching.

## Tokenizer, Embeddings, and LM Head

Borrow the tokenizer for v1, but use plain token and position embeddings unless an existing compatible embedding module is already available. The goal is to validate the JEPA block, not to build tokenization infrastructure.

Initial contract:

```text
input_ids:      (B, L)
labels:         (B, L)
valid_mask:     (B, L), optional loss mask
```

Input embeddings:

```text
token_embedding:    (vocab_size, D)
position_embedding: (max_length, D)
h_0 = token_embedding[input_ids] + position_embedding[position_ids]
```

The LM head projects the final residual stream to vocabulary logits:

```text
logits = lm_head(final_norm(h_depth))
```

The LM loss is ordinary next-token cross entropy:

```text
loss_lm = cross_entropy(logits[:, :-1], labels[:, 1:])
```

Do not use same-token reconstruction for the LM head in the first pass.

Weight tying is optional:

```python
if tie_weights:
    lm_head.weight = token_embedding.weight
```

Only tie when both shapes are `(vocab_size, D)`. Do not add projection glue just to force tying.

## Why This Differs From The Existing Layer Model

The existing `LayerModel` follows the standard student/teacher text-JEPA shape:

```text
masked input -> context tower -> predictor -> target latents
full input   -> EMA target tower -> target latents
```

Intertwined H-JEPA instead makes predictions **inside the depth recurrence**:

```text
layer l post-attention residual -> predictor delta -> next residual
future layer post-attention residual -> EMA compressor -> target for previous predictor
```

Consequences:

- the predictor cannot be thrown away after training
- training and inference must use the same delta-injected residual dynamics
- the target state depends on a future internal activation from the same forward pass
- the model needs explicit storage of post-attention states and compressed states at every layer
- the JEPA loss supervises the predictor as a true delta function: `d_l ~= target_z_l - sg(z_l)`

## First Tensor Contract

Use dense full-sequence tensors for v1.

```text
h_l:             (B, L, D)
h_l_post_attn:   (B, L, D)
z_l:             (B, L, K)
d_l:             (B, L, K)
Proj_l(...):     (B, L, D)
logits:          (B, L, vocab_size)
```

Where:

- `B` is batch size
- `L` is sequence length
- `D` is residual width
- `K` is JEPA compressed width

## Error Handling Policy

Do not add broad defensive validation in the first pass. Let PyTorch report normal tensor shape, dtype, and matmul errors.

Manual checks should be limited to architecture invariants where a later failure would be confusing:

```text
depth >= 2
delta_l, z_l, and target_z_l have the same shape in JEPA loss
valid_mask selects at least one position if it is used
```

## Forward Pass Order

Because `loss_l` needs the next layer's post-attention state, the forward pass should first compute and store all student states, then compute teacher targets and losses.

Suggested structure:

```python
h = token_embeddings(input_ids) + position_embeddings(position_ids)
post_attn_states = []
compressed = []
deltas = []

for l in range(depth):
    h_normed = attn_norms[l](h)
    attn_out = attentions[l](h_normed)
    h_post_attn = h + attn_out

    z_l = compressors[l](ce_norms[l](h_post_attn))
    d_l = predictors[l](z_l)
    h = h_post_attn + projectors[l](z_l + d_l)

    post_attn_states.append(h_post_attn)
    compressed.append(z_l)
    deltas.append(d_l)

logits = lm_head(final_norm(h))

jepa_losses = []
for l in range(depth - 1):
    with torch.no_grad():
        target_z_l = ema_compressors[l + 1](
            ce_norms[l + 1](post_attn_states[l + 1])
        )

    target_delta = target_z_l.detach() - compressed[l].detach()
    jepa_losses.append(mse(deltas[l], target_delta))

loss_main = next_token_cross_entropy(logits, labels, valid_mask)
loss = loss_main + lambda_jepa * mean(jepa_losses)
```

## JEPA Loss

For v1, use plain MSE only:

```text
loss_jepa_l = MSE(d_l, target_z_l.detach() - z_l.detach())
```

Default:

```text
lambda_jepa = 0.1
```

Do not use SIGReg in the initial pass.

## EMA Contract

EMA copies are per-layer teacher compressors only:

```text
CEbar_l <- momentum * CEbar_l + (1 - momentum) * CE_l
```

First-pass EMA scope:

```text
EMA includes:  compressors CE_l
EMA excludes:  predictors P_l
EMA excludes:  projectors Proj_l
EMA excludes:  attention weights
EMA excludes:  embeddings
EMA excludes:  LM head
```

Use a clean helper:

```python
def make_ema_copy(module: nn.Module) -> nn.Module:
    ema = copy.deepcopy(module)
    ema.requires_grad_(False)
    ema.eval()
    return ema
```

Training order:

```text
1. student forward
2. no_grad target computation with CEbar_{l+1}
3. total loss
4. backward
5. optimizer.step()
6. update_ema()
```

Do not put EMA modules in AdamW.

## Minimal Implementation Layout

Keep the first implementation compact. Prefer one main model file plus focused tests.

Suggested local files:

```text
intertwined_hjepa.py
test_intertwined_hjepa_shapes.py
test_intertwined_hjepa_training_step.py
```

The model file should contain:

- config dataclass
- attention block pieces
- compressor
- predictor
- projector
- intertwined model
- JEPA delta loss
- next-token loss
- EMA update helper

## Milestones

1. Implement one model file with:
   - token + position embeddings
   - pre-norm causal self-attention per layer
   - compressor / predictor / projector
   - LM head
   - EMA compressors

2. Add shape and detach tests:
   - block output shapes
   - JEPA loss shape contract
   - no-grad EMA target path
   - final layer excluded from JEPA loss

3. Add one training-step smoke test:
   - forward
   - backward
   - optimizer step
   - EMA update

4. Only then consider:
   - better compressor shapes
   - scaling laws
   - masked JEPA positions
   - extra regularization
