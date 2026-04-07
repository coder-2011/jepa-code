# Example Masked-Span Flow

"The quick brown fox jumps over the lazy dog"

Masking decision:

- Visible (context): "The quick brown ___ ___ over the lazy dog"
- Hidden (target): "fox jumps"

Simultaneously:

## Context encoder

Input:

- "The quick brown [MASK] [MASK] over the lazy dog"

Output:

- `Sx` - representations of the full sequence
- with `[MASK]` tokens at hidden positions

## Target encoder

EMA copy, no gradients.

Input:

- "The quick brown fox jumps over the lazy dog"
- full sequence, nothing hidden

Output:

- `Sy` - representations including "fox" and "jumps"

## Predictor

Input:

- `Sx`
- positional hints for positions 3 and 4

Output:

- `Shat_y` - predicted representations for positions 3 and 4

## Loss

`distance(Shat_y, Sy at positions 3 and 4)`

= how wrong was the prediction of "fox jumps" representations
