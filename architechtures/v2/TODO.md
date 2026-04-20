# TODO

- Use JEPA only on a subset of layers.
  Example: every 3rd layer, or only upper-half layers. That gives depth room to specialize.

- Predict farther than one token.
  Even 2-4 token future targets, or mixed horizons, could force less trivial structure.

- Make teacher targets come from a different view.
  Examples: masked future chunk, strided view, pooled future window, or a higher-level summary target instead of same-token-position next-step compression.

- Predict a chunk summary, not pointwise tokenwise `z`.
  A pooled or low-rate target could encourage abstraction instead of token-local imitation.

- Stop using the exact same JEPA contract at every layer.
  Lower layers could do local structure; upper layers could do chunk/future-summary prediction.

- Consider making JEPA target post-transition states rather than compressed same-layer post-attn states.
  Right now the target may still be too close to the student’s immediate representation family.

- Strong guess: the next real gain probably will not come from tuning coefficients.
  It is more likely to come from making the JEPA task harder and less redundant with the LM task, while applying it to fewer layers.
