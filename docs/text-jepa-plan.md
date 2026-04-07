# Text JEPA Implementation Plan

## Goal

Build a clean text-first Joint-Embedding Predictive Architecture system from scratch, without depending on the legacy `jepa/` package.

The target system is:

- encoder-style, not autoregressive
- span-masked, with target spans replaced by `[MASK]` tokens in the context input
- trained by latent prediction, not token reconstruction
- based on a trainable context encoder, an EMA-updated target encoder, a predictor, and a latent loss

This document is the execution plan.

The companion deep-dive flow document is:

- `docs/text-jepa-flow.md`

## What We Are Building

### Canonical core

The architecture we are committing to is:

1. Context encoder
2. Target encoder
3. Predictor
4. Latent loss
5. EMA update
6. Span masking and target extraction

### Explicit non-goals for v1

We are **not** building these first:

- autoregressive next-token loss
- LeJEPA SIGReg
- multi-view `Text -> Code` JEPA as the primary training path
- decoder-only causal attention as the default baseline
- broad framework abstractions before the core contracts are stable

## Evidence Base

### Primary sources

- I-JEPA paper:
  `JEPA.pdf`
- LeJEPA paper:
  `LeJEPA.pdf`
- LLM-JEPA paper:
  `LLM-JEPA Large Language Models.pdf`

### Online sources used

- data2vec:
  <https://arxiv.org/abs/2202.03555>
- data2vec 2.0:
  <https://arxiv.org/abs/2212.07525>
- BYOL:
  <https://arxiv.org/abs/2006.07733>
- public `llm-jepa` repo:
  <https://github.com/galilai-group/llm-jepa>

### Key takeaways from the evidence

1. I-JEPA establishes the core JEPA template:
   target encoder, context encoder, predictor, latent regression loss, EMA teacher.
2. data2vec extends the same idea to language:
   predict latent representations of the full input from a masked view with a self-distillation setup.
3. data2vec 2.0 shows the efficiency lesson:
   masking policy and teacher-target construction strongly affect runtime and practicality.
4. BYOL is the clean precedent for online network plus EMA target network.
5. Original LLM-JEPA is not the exact system we want:
   it mixes JEPA with autoregressive loss and whole-view embeddings.
6. The current public `llm-jepa` repo now includes a `Semantic Tube Prediction` path with random spans, which is closer to our direction, but still not a complete substitute for our design.

## Architecture Decision

### Chosen v1 system

We will build a **bidirectional text JEPA encoder system** with these semantics:

- input text is tokenized to a fixed padded length `L`
- a subset of contiguous spans is chosen as the target set
- the context input retains full length, but target tokens are replaced by `[MASK]`
- the context encoder processes the masked full sequence
- the target encoder processes the unmasked full sequence
- the predictor receives context states plus target-position queries
- the loss compares predicted latent states against stop-gradient target latent states

### Why this is the right baseline

This is the cleanest text analogue of I-JEPA under the user's stated constraints:

- `[MASK]` tokens remain in sequence
- target latents are fixed targets
- target encoder is EMA only
- loss flows only through predictor and context encoder

It also matches the highest-signal part of data2vec:

- masked input on the student side
- full input on the teacher side
- latent regression, not token classification

## Proposed Code Structure

Create a fresh package, independent of `jepa/`.

Suggested layout:

```text
docs/
  text-jepa-plan.md
  text-jepa-flow.md

src/
  text_jepa/
    __init__.py
    config.py
    tokenization.py
    masking.py
    batching.py
    utils/
      tensor_ops.py
      ema.py
      attention.py
    models/
      embeddings.py
      encoder_block.py
      encoder.py
      context_encoder.py
      target_encoder.py
      predictor.py
      text_jepa.py
    losses/
      latent_loss.py
    train/
      step.py
      loop.py
      metrics.py

tests/
  test_masking.py
  test_batching.py
  test_encoder_shapes.py
  test_predictor_shapes.py
  test_loss.py
  test_ema.py
  test_training_step.py
```

## Core Contracts

These are the first contracts we should lock and test before deeper implementation work.

### Batch contract

Each training batch should provide:

- `input_ids_full: (B, L)`
- `input_ids_ctx: (B, L)`
- `attention_mask: (B, L)`
- `target_mask: (B, L)` bool
- `target_positions: (B, T_max)` int
- `target_valid_mask: (B, T_max)` bool

### Model contract

The full model forward should return:

- `context_states: (B, L, D)`
- `target_states: (B, L, D)` or detached gathered targets
- `predicted_target_states: (B, T_max, D)`
- `target_target_states: (B, T_max, D)`
- `target_valid_mask: (B, T_max)`

### Loss contract

Loss consumes:

- predictions `(B, T_max, D)`
- targets `(B, T_max, D)`
- validity mask `(B, T_max)`

and returns:

- scalar latent loss

### Training contract

Backpropagation path:

- loss -> predictor -> context encoder

No gradient path:

- loss -X-> target encoder

Update rule after optimizer step:

- `theta_target <- tau * theta_target + (1 - tau) * theta_context`

## Workstreams

## 1. Tokenization and span masking

Deliverables:

- tokenizer wrapper
- `[MASK]` token handling
- contiguous span sampler
- deterministic padding and truncation
- `target_positions` extraction

Acceptance criteria:

- masked positions are reproducible under fixed seed
- `input_ids_ctx` and `input_ids_full` stay aligned at length `L`
- gathered target positions decode to the original unmasked text

## 2. Context encoder

Deliverables:

- token embedding
- positional embedding
- encoder block stack
- full-sequence masked-input forward pass

Acceptance criteria:

- output shape is always `(B, L, D)`
- layer-wise shape invariants hold under variable batch sizes
- `[MASK]` positions remain valid latent slots, not dropped tokens

## 3. Target encoder

Deliverables:

- same architecture as the context encoder
- copied initialization
- EMA update utility
- no-grad forward path

Acceptance criteria:

- identical tensor shape to context encoder
- target parameters never receive optimizer gradients
- EMA updates numerically change target parameters after each step

## 4. Predictor

Deliverables:

- query construction from target positions
- query self-attention
- cross-attention into context states
- final prediction head to `D`

Acceptance criteria:

- input shape `(B, T_max, D)` for query stream
- memory shape `(B, L, D)` from context stream
- output shape `(B, T_max, D)`
- padded target slots do not contribute to the loss

## 5. Loss system

Deliverables:

- masked MSE baseline
- optional cosine and normalized-MSE ablations
- target gather utility

Acceptance criteria:

- only valid target positions contribute
- loss is zero when predictions equal targets
- stop-gradient target path is verified

## 6. Training step

Deliverables:

- one-step forward-backward-update function
- optimizer step
- EMA step
- basic metrics

Acceptance criteria:

- no runtime shape mismatches on synthetic data
- gradient norms are non-zero for context encoder and predictor
- gradient norms are zero for target encoder

## 7. Tests and smoke runs

Deliverables:

- unit tests for all shape contracts
- tiny synthetic smoke run
- overfit-one-batch test

Acceptance criteria:

- test suite passes locally
- one-batch training reduces latent loss
- EMA target drift behaves as expected

## Milestone Plan

## Milestone 0: Contracts before code

Write and freeze:

- notation
- tensor shapes
- forward signatures
- batch schema

Exit condition:

- no unresolved ambiguity in the flow document for the v1 system

## Milestone 1: Data and masking

Implement:

- tokenizer wrapper
- span sampler
- batch builder

Exit condition:

- can construct `input_ids_full`, `input_ids_ctx`, `target_positions`, and masks for a batch

## Milestone 2: Encoders

Implement:

- embeddings
- encoder blocks
- context encoder
- target encoder wrapper

Exit condition:

- both encoders return `(B, L, D)` on synthetic batches

## Milestone 3: Predictor

Implement:

- query builder
- cross-attention predictor
- padded target handling

Exit condition:

- predictor returns `(B, T_max, D)` and passes shape tests

## Milestone 4: Loss and training step

Implement:

- latent loss
- batched gather
- stop-gradient handling
- EMA update

Exit condition:

- single training step runs end to end

## Milestone 5: Minimal training loop

Implement:

- trainer loop
- logging
- checkpointing

Exit condition:

- can train on a tiny corpus for several steps without failure

## Milestone 6: Ablations

Add:

- cosine loss
- normalized MSE
- alternate predictor depths
- alternate span samplers

Exit condition:

- can compare core design choices without rewriting the base system

## Recommended Defaults for v1

These are proposed defaults, not canonical paper constants.

- tokenizer: BPE or WordPiece tokenizer with explicit `[MASK]`
- architecture: encoder-only Transformer
- sequence length `L`: 256 or 512 for early experiments
- hidden size `D`: 512 or 768
- heads `H`: `D / 64`
- encoder depth: 6 to 12 layers for the first working version
- predictor depth: 2 to 4 blocks
- target span count per sample: 1 to 4
- total masked ratio: start around 15 to 30 percent
- loss: masked MSE over target positions
- EMA coefficient `tau`: start near `0.99` to `0.999`

## Main Open Decisions

## 1. Encoder family

Choice:

- encoder-only Transformer

Alternative:

- decoder-only LLM with custom attention masks

Recommendation:

- use encoder-only first

Reason:

- it directly matches masked-span latent prediction without importing autoregressive assumptions

## 2. Predictor memory

Choice:

- cross-attend to full `Sx`

Alternative:

- cross-attend only to non-target positions

Recommendation:

- start with full `Sx`, then ablate

Reason:

- `[MASK]` positions in `Sx` do not leak target tokens; they are learned latent placeholders

## 3. Loss metric

Choice:

- MSE

Alternatives:

- cosine loss
- normalized MSE
- L2 norm

Recommendation:

- MSE first, cosine as an ablation

Reason:

- MSE is closer to I-JEPA and data2vec style latent regression, while cosine was preferred in the original whole-view LLM-JEPA setup

## 4. Span sampler

Choice:

- contiguous spans

Alternatives:

- token-level masking
- random scattered subsets
- syntax-aware spans

Recommendation:

- contiguous spans first

Reason:

- they are the cleanest 1D analogue of image target blocks

## Risks

## 1. Architectural drift

Risk:

- mixing encoder JEPA, whole-view LLM-JEPA, and LeJEPA into one unstable design

Mitigation:

- keep v1 as raw text JEPA only

## 2. Silent shape bugs

Risk:

- target positions, padding, and gather logic will fail silently if not tested aggressively

Mitigation:

- shape-first unit tests

## 3. Target collapse or leakage

Risk:

- accidental gradient flow into the target encoder
- accidental use of unmasked targets on the context branch

Mitigation:

- explicit no-grad wrappers
- explicit tensor assertions

## 4. Overengineering too early

Risk:

- spending time on generalized framework code before the base path works

Mitigation:

- build the shortest end-to-end training path first

## Immediate Next Step

After these docs are accepted, the next implementation artifact should be the data contract:

- a span masking module
- a batch schema
- a synthetic batch generator

That gives us the first executable foundation for everything else.

## References

- I-JEPA paper:
  `JEPA.pdf`
- LeJEPA paper:
  `LeJEPA.pdf`
- LLM-JEPA paper:
  `LLM-JEPA Large Language Models.pdf`
- data2vec:
  <https://arxiv.org/abs/2202.03555>
- data2vec 2.0:
  <https://arxiv.org/abs/2212.07525>
- BYOL:
  <https://arxiv.org/abs/2006.07733>
- public `llm-jepa` repo:
  <https://github.com/galilai-group/llm-jepa>
