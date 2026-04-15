# Intertwined H-JEPA Plan

## Purpose

This document defines the first implementation plan for an **Intertwined Hierarchical JEPA** architecture.

The design goal is not to build a standard JEPA pretraining head. The first pass is a compact residual-stream model in which each non-final layer learns a JEPA-style predictive delta toward a one-layer-future EMA teacher state, and that delta is added back into the residual stream during both training and inference.

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

- `x`: borrowed token ids
- `e_0`: learned token plus position embeddings
- `h_l`: residual stream entering JEPA block `l`
- `z_l`: compressed representation of `h_l`
- `CE_l`: online compressor at layer `l`
- `CEbar_l`: EMA copy of compressor `CE_l`
- `P_l`: predictor at layer `l`
- `d_l`: predicted delta at layer `l`

The online path is:

```text
x -> token_embedding + position_embedding -> h_0
h_l -> compressor CE_l -> z_l
z_l -> predictor P_l -> d_l
h_{l+1} = h_l + Proj_l(z_l + d_l)
h_depth -> final_norm -> lm_head -> token logits
```

The teacher target for non-final layer `l` is:

```text
target_z_l = stopgrad(target_project_l(CEbar_{l+1}(h_{l+1})))
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

The LM head is part of the v1 model. It gives the architecture a direct token-level output surface while the intertwined JEPA losses shape the internal residual dynamics.

## Autoregressive Generation Loop

The intended inference loop is autoregressive:

```text
1. Start with an input prefix.
2. Run the full JEPA stack.
3. Use the LM head to produce logits for one next token.
4. Pick/sample that token.
5. Append it to the sequence.
6. Run the model again to produce the following token.
```

So each generation pass produces one new token. Training can still use teacher forcing over a full sequence for efficiency:

```text
training logits:   (B, L, vocab_size)
generation logits: (B, vocab_size) for the final/current position
```

The first implementation does not need KV caching or incremental-state caching. It can rerun the available prefix each generation step.

## Borrowed Tokenizer, Embeddings, and LM Head

Borrow the tokenizer for v1, but use plain token and position embeddings unless an existing compatible embedding module is already available. The goal is to validate the JEPA block, not to build tokenization infrastructure.

Initial tokenizer contract:

```text
input_ids:      (B, L)
labels:         (B, L)
valid_mask:     (B, L), optional loss mask
```

Input embeddings are plain token plus position embeddings:

```text
token_embedding:    (vocab_size, D)
position_embedding: (max_length, D)
h_0 = token_embedding[input_ids] + position_embedding[position_ids]
```

The LM head should project the final residual stream to vocabulary logits:

```text
logits = lm_head(final_norm(h_depth))
training logits:   (B, L, vocab_size)
generation logits: (B, vocab_size) from the final/current position
```

Weight tying is optional. For v1, prefer tying `lm_head.weight` to `token_embedding.weight` if `D` matches the embedding width and the implementation stays simple.

Weight tying contract:

```python
if tie_weights:
    lm_head.weight = token_embedding.weight
```

Only tie weights when:

```text
token_embedding.weight.shape == (vocab_size, D)
lm_head.weight.shape == (vocab_size, D)
```

Do not add projection glue just to force weight tying in the first pass. If borrowed embeddings do not expose a compatible `token_embedding.weight`, use an untied LM head:

```python
lm_head = nn.Linear(D, vocab_size, bias=False)
```

With tied weights, the embedding table receives gradients from both input lookup usage and the LM loss. That is expected. The embedding table is still not part of EMA.

Training can combine losses:

```text
loss = lambda_jepa * loss_jepa + lambda_lm * loss_lm
```

Start with:

```text
lambda_jepa = 0.1
lambda_lm = 1.0
```

Then ablate:

```text
JEPA only
LM only
JEPA + LM
```

The LM loss should initially be ordinary next-token cross entropy:

```text
loss_lm = cross_entropy(logits[:, :-1], labels[:, 1:])
```

Do not use same-token reconstruction for the LM head in the first pass. The LM head predicts the next token.

For later masked-input experiments, keep the contracts explicit:

- JEPA can use masked or unmasked inputs depending on the experiment.
- LM loss should only be computed where labels are meaningful and not padding.
- If the model input contains `<mask>`, decide whether LM predicts all next tokens or only masked/reconstruction positions. Baseline answer: next-token LM on unmasked simple text first, masked JEPA second.

## Why This Differs From The Existing Layer Model

The existing `LayerModel` follows the standard student/teacher text-JEPA shape:

```text
masked input -> context tower -> predictor -> target latents
full input   -> EMA target tower -> target latents
```

Intertwined H-JEPA instead makes predictions **inside the depth recurrence**:

```text
layer l residual -> predictor delta -> next residual
future layer residual -> EMA compressor -> target for previous predictor
```

Consequences:

- The predictor cannot be thrown away after training.
- Training and inference must use the same delta-injected residual dynamics.
- The target state depends on a future internal activation from the same forward pass.
- The model needs explicit storage of residual states and compressed states at every layer.
- The JEPA loss supervises the predictor as a true delta function: `d_l ~= target_z_l - sg(z_l)`.

## First Tensor Contract

Use dense full-sequence tensors for v1. Sparse target-token prediction can be added later.

```text
h_l:      (B, L, D)
z_l:      (B, L, K)
d_l:      (B, L, K)
y_l:      (B, L, K)
z_l+d_l:  (B, L, K)
proj_l:   (B, L, D)
```

Where:

- `B` is batch size.
- `L` is sequence length.
- `D` is residual-stream width.
- `K` is JEPA predictive-space width.
- `Proj_l(z_l + d_l)` maps the enriched compressed representation back into residual width.

The baseline should set `K = D` only if that keeps the implementation simpler. Otherwise, prefer an explicit compressed JEPA space `K < D` so the plan's "compressed representation" has architectural reality.

## Error Handling Policy

Do not add broad defensive validation in the first pass. Let PyTorch report normal tensor shape, dtype, and matmul errors.

Manual checks should be limited to architecture invariants where a later failure would be confusing:

```text
depth >= 2
delta_l, z_l, and target_z_l have the same compressed dimension K
valid_mask selects at least one position if it is used in a masked loss
EMA compressor structure matches the student compressor structure during EMA update
```

Avoid checking every input rank, every optional key, or every config field unless someone would realistically need that error message to debug the architecture.

## Forward Pass Order

Because `loss_l` needs the layer `l + 1` residual state, the forward pass should first compute and store all student states, then compute teacher targets and losses.

Suggested structure:

```python
h = token_embeddings(input_ids) + position_embeddings(position_ids)
states = []
compressed = []
deltas = []
targets = []

for l in range(depth):
    states.append(h)
    z_l = compressors[l](h)

    if l < depth - 1:
        d_l = predictors[l](z_l)
        projected_update = projectors[l](z_l + d_l)
        h = h + projected_update

        compressed.append(z_l)
        deltas.append(d_l)

states.append(h)

jepa_losses = []
for l in range(depth - 1):
    with torch.no_grad():
        target_z_l = ema_compressors[l + 1](states[l + 1])
        target_z_l = target_projectors[l](target_z_l)
    target_delta = target_z_l - compressed[l].detach()
    jepa_losses.append(delta_loss_fn(deltas[l], target_delta))

logits = lm_head(final_norm(h))
loss_main = next_token_cross_entropy(logits, labels, valid_mask)
loss = loss_main + weighted_sum(jepa_losses)
```

Important implementation note: the first pass is only the JEPA residual transform.

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
EMA excludes:  token embeddings and LM head
```

Update timing:

```text
forward -> loss -> backward -> optimizer.step() -> ema_update()
```

Gradient rule:

- Gradients flow through compressors, predictors, projectors, embeddings, and the LM head.
- Gradients do not flow through EMA compressors.
- Target projection modules are only needed if adjacent compressed dims differ. For first pass with one shared `K`, use `Identity`.

The initial EMA copy should be exact:

```text
momentum = 0.0 for initialization
```

Implementation rules:

```text
1. Initialize CEbar_l as an exact copy of CE_l before training.
2. Exclude CEbar_l parameters from the optimizer.
3. Compute target_z_l under torch.no_grad().
4. Run one backward pass on total loss.
5. Run optimizer.step().
6. Run update_ema() after optimizer.step().
```

Do not call `update_ema()` before the optimizer step, and do not update EMA during forward.

Use a helper for EMA construction:

```python
def make_ema_copy(module: nn.Module) -> nn.Module:
    ema = copy.deepcopy(module)
    ema.requires_grad_(False)
    ema.eval()
    return ema
```

Then:

```python
self.ema_compressors = make_ema_copy(self.compressors)
```

This is cleaner than manually looping over parameters to call `requires_grad_(False)`.

## Losses

Start with:

```text
loss_jepa_l = MSE(d_l, stopgrad(target_z_l) - stopgrad(z_l))
```

No normalized MSE, cosine loss, SIGReg, VICReg, variance floor, or covariance penalty in the initial pass.

The initial training loss is:

```text
L_total =
  L_main
  + sum_l lambda_l * MSE(d_l, stopgrad(target_z_l) - stopgrad(z_l))
```

Where:

- `L_main` is next-token cross entropy from the final LM head.
- Initial `lambda_jepa = 0.1`.
- `lambda_jepa` should warm up from zero after the LM path starts learning for real training. Shape tests can set warmup to zero.
- SIGReg is intentionally not used in the initial pass.

Code form:

```python
target_delta = target_z_l.detach() - z_l.detach()
loss_jepa_l = F.mse_loss(delta_l, target_delta)
```

Masked form, only if a valid mask is supplied:

```python
error = (delta_l - target_delta).pow(2)
mask = valid_mask.unsqueeze(-1).to(error.dtype)
loss_jepa_l = (error * mask).sum() / (mask.sum() * error.shape[-1])
```

## Delta Injection

The first pass always uses direct residual injection:

```text
h_{l+1} = h_l + Proj_l(z_l + d_l)
```

## Masking Policy

For the first implementation, do not make masking a dependency of the architecture.

Recommended phases:

1. Dense synthetic smoke test: no masking, tiny tensors, verify shape/loss/backward/EMA.
2. Borrowed-tokenizer text smoke: full sequence, no JEPA masking.
3. Masked/span loss only after the dense version is stable.

This preserves the JEPA principle that the model should predict unavailable or future latent information, while keeping early debugging tractable.

## Proposed Code Layout

Keep this outside `jepa/` and `lejepa/`.

Prefer a compact implementation. This architecture is experimental, and too many small files would hide the training contract across module boundaries. Long but coherent files are acceptable here.

Recommended v1 layout:

```text
src/text_jepa/models/intertwined_hjepa.py
tests/test_intertwined_hjepa_shapes.py
tests/test_intertwined_hjepa_training_step.py
```

Optional later:

```text
docs/intertwined-hjepa-flow.md
src/text_jepa/models/intertwined_block.py
src/text_jepa/losses/intertwined_loss.py
scripts/train_intertwined_hjepa.py
intertwined-hjepa-default.yaml
```

`intertwined_hjepa.py` should initially contain:

- `IntertwinedConfig`
- `IntertwinedBlock`
- `SimpleCompressor`
- `DeltaPredictor`
- `IntertwinedHJEPA`
- delta loss helper
- next-token loss helper
- EMA update method

Only split files later if one file becomes actively painful to test or edit.

## Development Breakdown

### Step 0: Lock The Contract In Code Comments

Goal:

- Put the exact training-step equation near the top of `intertwined_hjepa.py`.

Contract:

```text
h_l = residual state entering JEPA block l
z_l = CE_l(h_l)
d_l = Pred_l(z_l)
h_{l+1} = h_l + Proj_l(z_l + d_l)

y_l = sg(CEbar_{l+1}(h_{l+1}))
L_jepa_l = MSE(d_l, target_z_l - sg(z_l))
```

Acceptance criteria:

- Anyone opening the file can see the math before the implementation.

### Step 1: Borrow Input Pipeline

Goal:

- Avoid tokenizer work in the first pass.

Implementation:

- Reuse an existing tokenizer/embedding setup or feed synthetic `input_ids`.
- Keep the JEPA model contract at tensor level: `input_ids`, optional `labels`, optional `valid_mask`.

Acceptance criteria:

- Model tests can run without implementing a tokenizer.
- Tokenizer-specific behavior is not part of the base JEPA block tests.

### Step 2: One Compact Model File

Goal:

- Implement the full model skeleton in one file.

Contents:

- config dataclass
- plain token and position embeddings
- optional LM-head weight tying when token embedding and LM-head shapes match
- compact base JEPA block with compressor, predictor, and projector
- compressors `CE_l`
- predictors `Pred_l`
- projectors `Proj_l`
- final norm
- LM head
- EMA compressor copies

Acceptance criteria:

- `forward(input_ids, labels=None, valid_mask=None, return_diagnostics=True)` runs.
- Returns `logits`, `final_states`, and diagnostics even without labels.

### Step 3: Student Forward Without Loss

Goal:

- Get the residual recurrence exactly right before adding teacher targets.

Forward outputs:

```text
states:         list[(B, L, D)]
z:              list[(B, L, K)]
deltas:         list[(B, L, K)]
training logits:   (B, L, vocab_size)
generation logits: (B, vocab_size)
```

Acceptance criteria:

- Shapes are correct for depth >= 2.
- Direct residual injection changes the state at every non-final block.

### Step 4: EMA Copies And Target Computation

Goal:

- Add target branch without gradients.

Implementation:

- EMA copies of compressors only.
- Initial hard copy from compressors.
- `make_ema_copy()` helper using `deepcopy`, module-level `requires_grad_(False)`, and `eval()`.
- `update_ema()` method on the model.
- `compute_targets(states)` helper.

Acceptance criteria:

- Targets are computed under `torch.no_grad()`.
- Target tensors have shape `(B, L, K)`.
- EMA compressor parameters receive no gradients from loss.
- EMA compressors are not in the optimizer.
- EMA changes only after explicit `update_ema()` after `optimizer.step()`.

### Step 5: Losses

Goal:

- Add the exact total training loss.

Losses:

```text
L_main = next-token CE(logits, labels)
L_jepa_l = MSE(delta_l, target_z_l - stopgrad(z_l))
L_total = L_main + sum(lambda_l * L_jepa_l)
```

Acceptance criteria:

- One `.backward()` computes gradients.
- Compressors, predictors, projectors, embeddings, and LM head can receive gradients.
- EMA compressors do not receive gradients.

### Step 6: Warmup Controls

Goal:

- Make the training dynamics tunable from the beginning.

Implementation:

- Config fields:
  - `lambda_jepa`
  - `jepa_warmup_steps`
- Forward accepts optional `step`.
- Effective weights are linearly warmed up from zero.

Acceptance criteria:

- At step 0, JEPA can be effectively off.
- After warmup, configured weights are reached.
- Diagnostics report effective weights.

Default:

```text
lambda_jepa = 0.1
```

### Step 7: Diagnostics

Goal:

- Make collapse and delta behavior visible before scaling.

Diagnostics:

- total loss
- main loss
- per-layer JEPA loss
- per-layer `z` variance
- per-layer delta norm
- per-layer target-delta norm
- prediction/target norm ratio
- approximate effective rank if cheap

Acceptance criteria:

- Diagnostics are returned as detached CPU-friendly tensors/floats.
- Tests verify keys exist and are finite.

### Step 8: Tiny Training Loop Test

Goal:

- Prove the system can take optimizer and EMA steps.

Test:

- Tiny vocab.
- Tiny sequence length.
- Depth 3.
- Hidden dim 32 or smaller.
- Run 2 to 5 AdamW steps.

Acceptance criteria:

- Loss is finite.
- No NaNs.
- Some student parameter changes.
- Some EMA parameter changes after `update_ema()`.
- LM-only overfit path can reduce loss on a few strings.

### Step 9: Ablation Switches

Goal:

- Support the core scientific comparisons without rewriting the model.

Config switches:

- `use_jepa_loss`
- `use_lm_loss`

Key ablations:

```text
LM only
JEPA only
LM + JEPA with residual delta injection
```

Acceptance criteria:

- All four modes run.
- Direct residual injection is always active in the first pass.
 
### Step 10: Only Then Add Scripted Training

Goal:

- Avoid trainer complexity until the model is mechanically proven.

Deliverable:

- `scripts/train_intertwined_hjepa.py`, only after the tests above pass.

Acceptance criteria:

- Runs a tiny local text file.
- Logs the diagnostics.
- Saves a small checkpoint.

## Implementation Milestones

### Milestone 1: Architectural Skeleton

Deliverables:

- `IntertwinedBlock` implemented as the compact base JEPA block inside `intertwined_hjepa.py` for v1.
- `IntertwinedHJEPA` with plain token/position embeddings, compressors, EMA compressors, predictors, projectors, and LM head.
- Dense forward pass returning intermediate diagnostics.

Acceptance criteria:

- Forward returns `loss`, `loss_jepa`, optional `loss_lm`, `logits`, `final_states`, `predictions`, and `targets`.
- Only core architecture invariants are checked manually; ordinary tensor mistakes can fail through PyTorch.
- Final-layer JEPA loss is skipped.

### Milestone 1.5: Simple Tokenizer and LM Surface

Deliverables:

- Character-level tokenizer with deterministic encode/decode.
- Plain token and position embeddings.
- LM head on the final residual stream.
- Next-token cross-entropy helper.

Acceptance criteria:

- `logits.shape == (B, L, vocab_size)`.
- generation can select the final/current-position logits as `(B, vocab_size)`.
- LM loss ignores padding positions.
- A tiny LM-only overfit test can reduce loss on a few short strings.

### Milestone 2: EMA and Gradient Contract

Deliverables:

- EMA initialization and update utilities that support per-layer compressors.
- Tests proving teacher parameters receive no gradients.
- Tests proving optimizer updates compressor/predictor/projector parameters.

Acceptance criteria:

- `target.requires_grad` may be true as parameters, but no `.grad` is populated from the predictive loss.
- EMA compressor parameters change only after explicit EMA update.
- EMA compressor parameters are not included in the optimizer parameter groups.

### Milestone 3: Delta-Inference Contract

Deliverables:

- One forward mode used for both training and inference.
- Optional `return_loss=False` inference path that still applies predictor deltas.

Acceptance criteria:

- Predictor deltas are added during inference.
- JEPA projectors affect output states, proving predictors are structurally active.

### Milestone 4: Masked Text Integration

Deliverables:

- Use existing batch keys only after the base dense path works.
- Support dense full-sequence loss and masked-position loss.

Acceptance criteria:

- Existing Layer data pipeline can feed Intertwined H-JEPA with minimal adapter code.
- Loss can be restricted to masked target positions.

### Milestone 5: Tiny Training Smoke

Deliverables:

- Tiny synthetic training loop.
- One real tokenizer-backed smoke run.
- Logged metrics: total loss, main loss, per-layer JEPA loss, delta norm, target-delta norm, target norm, representation variance.

Acceptance criteria:

- Loss is finite.
- Backward works.
- EMA update works.
- A tiny run decreases loss over a few steps or at least does not diverge.

### Milestone 6: Collapse and Scale Checks

Deliverables:

- Variance/covariance diagnostics.
- Later optional regularizer experiment.

Acceptance criteria:

- Detect complete collapse: near-zero batch variance.
- Detect dimensional collapse: low effective rank or covariance concentration.
- Compare no-regularizer vs regularized variants later.

Minimum diagnostics:

- per-layer target variance
- per-layer prediction variance
- effective rank
- prediction/target norm ratio
- delta norm
- LM loss versus JEPA loss balance

## Open Design Questions

1. Should the EMA teacher be only `CEbar_{l+1}`, or should it include a larger learned target transform?
2. Should the teacher consume raw `h_{l+1}` or a normalized version of `h_{l+1}`?
3. Should `pred_l = z_l + d_l`, or should the target loss supervise only `d_l ~= target_z_l - stopgrad(z_l)`? Baseline answer: supervise the delta directly.
4. Should `z_l` be detached in the delta target? Baseline answer: yes for the target-delta term, while `z_l` still receives gradients through `Pred_l(z_l)`, `Proj_l(z_l + d_l)`, and `L_main`.
5. Should delta injection happen directly on the residual stream or after a future external mixing module? Baseline answer: directly on the residual stream for the first pass.
6. Should target projections be EMA copies too? Baseline answer: no. For first pass, use `Identity` by keeping one shared compressed dim `K`.

## Initial Recommendation

Build the first version as a small, explicit PyTorch model rather than trying to bend the existing `LayerModel` into this shape. Reuse utilities where they fit, especially EMA, losses, and reproducibility helpers, but keep the base JEPA block focused only on compression, delta prediction, and projection.

The first successful result should be boring:

```text
tiny model
borrowed tokenizer or random tokens
dense loss
finite backward
EMA update
predictor delta changes inference output
LM logits have the expected shape
```

Only after that should we integrate real span masking, real tokenizer batches, and optional collapse controls.
