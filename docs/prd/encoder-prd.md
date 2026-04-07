# Encoder PRD

## Status

Draft for the next text-JEPA implementation milestone.

## Goal

Take embedded token sequences of shape `(B, L, D)` and turn them into contextualized latent states of shape `(B, L, D)`.

This is the next step after:

1. tokenizer
2. masker
3. batching
4. embeddings

## Short Answer

The encoder should stay very simple in v1:

- one transformer encoder block
- self-attention
- feed-forward network
- residual connections
- layer norms
- stack `N` copies in a plain encoder module

That is enough to get the JEPA backbone started.

## Where This Fits

The flow is:

1. `input_ids_ctx: (B, L)`
2. embeddings -> `(B, L, D)`
3. encoder -> `(B, L, D)`
4. predictor / loss later

So the encoder is the first module that makes token representations context-aware.

Embeddings alone give:

- one learned vector per token

The encoder adds:

- left context
- right context
- interactions across the full sequence

## Input Contract

The encoder block should accept:

- `x: FloatTensor (B, L, D)`
- optional `attention_mask: (B, L)`

The full encoder should accept the same.

## Output Contract

The encoder block should return:

- `x_out: FloatTensor (B, L, D)`

The full encoder should return:

- `x_out: FloatTensor (B, L, D)`

So the encoder should preserve:

- batch size
- sequence length
- hidden size

## What We Need in v1

### 1. Encoder block

One block should do:

- layer norm
- multi-head self-attention
- residual add
- layer norm
- feed-forward network
- residual add

### 2. Encoder stack

The encoder module should:

- take embedded inputs `(B, L, D)`
- run them through `N` blocks
- return final hidden states `(B, L, D)`

## What We Do Not Need Yet

We do **not** need these in the first encoder milestone:

- causal masking
- cross-attention
- rotary embeddings
- KV cache logic
- flash-attention-specific optimizations
- predictor logic
- context/target encoder split
- EMA logic

The first encoder milestone is just:

- “does `(B, L, D)` go in and `(B, L, D)` come out correctly?”

## Recommended Files

```text
src/text_jepa/models/
  encoder_block.py
  encoder.py

tests/
  test_encoder_shapes.py
```

## Design Choices

## 1. Use bidirectional self-attention

This repo’s text JEPA design is encoder-style, not autoregressive.

So the encoder should use:

- full bidirectional attention over the sequence

not:

- causal masking

## 2. Keep hidden size fixed through the block

If input is `(B, L, D)`, output should stay `(B, L, D)`.

That keeps every later contract simple.

## 3. Use a standard FFN expansion

A simple choice is:

- `ffn_dim = 4 * D`

This is a standard transformer default and fine for v1.

## 4. Use PyTorch built-ins where possible

For the first pass, we should keep implementation simple and readable.

So using `nn.MultiheadAttention` or a clean equivalent is fine.

The goal is correctness and clarity, not special optimization yet.

## Pseudocode

### Encoder block

```python
class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x, attention_mask=None):
        x_norm = self.norm1(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
```

### Encoder stack

```python
class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, ffn_dim):
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        return self.final_norm(x)
```

## Minimal Config We Need

For the encoder milestone, the model config needs at least:

- `hidden_dim`
- `num_heads`
- `num_layers`
- `ffn_dim`

Reasonable v1 defaults:

```yaml
model:
  hidden_dim: 768
  num_heads: 12
  num_layers: 6
  ffn_dim: 3072
```

If we want to keep config even smaller, we can derive:

- `ffn_dim = 4 * hidden_dim`

and leave it out of YAML.

## Test Plan

Create `tests/test_encoder_shapes.py`.

Minimum tests:

1. encoder block preserves shape `(B, L, D)`
2. encoder stack preserves shape `(B, L, D)`
3. encoder accepts an attention mask without shape errors
4. repeated forward pass works on a tiny synthetic batch
5. invalid head count or bad dimensions fail clearly

## Example test pseudocode

```python
def test_encoder_block_preserves_shape():
    block = EncoderBlock(hidden_dim=8, num_heads=2, ffn_dim=32)
    x = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    y = block(x, attention_mask=attention_mask)

    assert y.shape == (2, 5, 8)
```

```python
def test_encoder_preserves_shape():
    encoder = Encoder(num_layers=2, hidden_dim=8, num_heads=2, ffn_dim=32)
    x = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    y = encoder(x, attention_mask=attention_mask)

    assert y.shape == (2, 5, 8)
```

## Acceptance Criteria

The encoder milestone is complete when:

- there is a simple `encoder_block.py`
- there is a simple `encoder.py`
- both preserve `(B, L, D)` shape
- both work with the current embedding output
- tests pass on small synthetic inputs

## Final Recommendation

Keep the first encoder extremely plain.

The right first goal is:

- stable bidirectional transformer encoder behavior
- correct shape contracts
- minimal moving parts

That will give us the clean backbone we need before building:

- context encoder
- target encoder
- predictor
- latent loss
