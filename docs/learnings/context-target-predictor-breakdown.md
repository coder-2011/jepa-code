# Context Encoder, Target Encoder, and Predictor Breakdown

For this text JEPA system, the three parts should be thought of as three different roles, not three unrelated models.

## Context Encoder

This is the main trainable encoder.

Input:

- full token sequence
- target spans replaced by `[MASK]`
- attention mask
- positional embeddings

Output:

- contextualized latent for every token position
- shape: $(B, L, D)$

Job:

- build a representation of the observed sequence
- include information from left and right context
- provide the latent memory that the predictor uses

In practice, this should just be a standard encoder-only Transformer:

- token embeddings
- positional embeddings
- $N$ Transformer encoder blocks
- final hidden states

So the context encoder is the "student" network.

## Target Encoder

This is the teacher network.

Input:

- full unmasked token sequence
- same attention mask
- same positional embeddings

Output:

- target latent for every token position
- shape: $(B, L, D)$

Job:

- produce the latent states that the model is trying to match at target positions

Important training rule:

- no gradients through this branch
- parameters updated only by EMA from the context encoder

So architecturally it is basically the same encoder as the context encoder, but operationally it is different:

- context encoder learns by backprop
- target encoder only tracks it slowly through EMA

You can think of it as:

- same structure
- different update rule
- different role in the computation graph

## Predictor

This is the module that turns context-side representations into predictions for masked target positions.

Input:

- context encoder output $S_x$ with shape $(B, L, D)$
- target position indices $p_{\mathrm{tgt}}$ with shape $(B, T_{\max})$
- optional target-position embeddings or learned query tokens

Output:

- predicted latent states only for the target positions
- shape: $(B, T_{\max}, D)$

Job:

- ask: "given the masked sequence representation, what should the teacher latent be at these masked positions?"

This is not another full encoder over the sequence.

It is a target-conditioned prediction head.

Best first design:

- one learned query seed vector
- add target positional embeddings to create target queries
- run a small Transformer-style predictor:
  - self-attention over target queries
  - cross-attention from target queries into full context states $S_x$
  - FFN
- output one $D$-dimensional latent per target token

So:

- context encoder creates memory
- predictor queries that memory to reconstruct target latents

## How They Fit Together

1. Context encoder sees masked sequence:
   - outputs $S_x$ with shape $(B, L, D)$

2. Target encoder sees full sequence:
   - outputs $S_y$ with shape $(B, L, D)$

3. Gather teacher targets at masked positions:
   - $S_{y,\mathrm{tgt}}$ with shape $(B, T_{\max}, D)$

4. Predictor uses $S_x$ with target positions:
   - outputs $\hat{S}_{y,\mathrm{tgt}}$ with shape $(B, T_{\max}, D)$

5. Loss compares:
   - $\hat{S}_{y,\mathrm{tgt}}$ vs $\operatorname{stopgrad}(S_{y,\mathrm{tgt}})$

## Simple Mental Model

- Context encoder = "understand the corrupted text"
- Target encoder = "define the correct latent answer"
- Predictor = "map context understanding into missing-span latent predictions"

## Most Concrete Implementation Wording

- Context encoder:
  a bidirectional Transformer encoder over masked text
- Target encoder:
  EMA copy of the same encoder over unmasked text
- Predictor:
  a smaller query-based Transformer that predicts latent states at masked positions from context memory

## Notes

- $B$ is batch size
- $L$ is sequence length
- $D$ is hidden size
- $T_{\max}$ is the padded maximum number of target positions in the batch
