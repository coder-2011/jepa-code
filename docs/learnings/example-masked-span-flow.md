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

- $S_x$ - representations of the full sequence
- with `[MASK]` tokens at hidden positions

## Target encoder

EMA copy, no gradients.

Input:

- "The quick brown fox jumps over the lazy dog"
- full sequence, nothing hidden

Output:

- $S_y$ - representations including "fox" and "jumps"

## Predictor

Input:

- $S_x$
- positional hints for positions $3$ and $4$

Output:

- $\hat{S}_y$ - predicted representations for positions $3$ and $4$

## Loss

$$
\operatorname{distance}\!\left(\hat{S}_y, S_y\big|_{\{3,4\}}\right)
$$

= how wrong was the prediction of "fox jumps" representations
