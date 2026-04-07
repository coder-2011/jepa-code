# Tokenizer and Masker PRD

## Status

Draft for the first text-JEPA implementation milestone.

## Objective

Design the first two text data components for the JEPA workstream:

1. a tokenizer wrapper built on Hugging Face `AutoTokenizer`
2. a random block masker for contiguous text spans

These should be the first stable data contracts because every later text-JEPA component depends on them:

- batch building
- target-position extraction
- context-input construction
- predictor query construction
- latent-loss masking

## User Requirements Captured

The current product requirements from discussion are:

- use a Hugging Face tokenizer rather than building one from scratch
- start with:
  `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`
- make the tokenizer model id configurable in YAML
- implement random block masking first
- keep the masking policy simple in v1
- default masking percentage should live in YAML
- create thorough tests before depending on these utilities in training code

## Research Summary

### 1. Hugging Face tokenizer loading

Hugging Face `AutoTokenizer.from_pretrained(...)` is the standard API for loading the tokenizer associated with a model repo. This is the correct mechanism for `Qwen/Qwen3-0.6B`.

Implication:

- we should not hardcode tokenizer classes
- configuration should store a model id string
- the wrapper should allow future tokenizer swaps without changing training code

### 2. Qwen3 compatibility note

The `Qwen/Qwen3-0.6B` model card shows tokenizer loading via `AutoTokenizer.from_pretrained(model_name)` and advises using a recent `transformers` release for Qwen3 support.

Implication:

- the implementation should pin or document a minimum `transformers` version that supports Qwen3
- tokenizer initialization errors should fail early with a clear message

### 3. Whole-word masking in Transformers is not the right primitive here

Hugging Face documents that `DataCollatorForWholeWordMask` relies on BERT-style `##` subword conventions and otherwise degrades to behavior similar to regular token masking.

Implication:

- we should not depend on `DataCollatorForWholeWordMask` for Qwen tokenization
- we need a custom masking module that is tokenizer-aware but not BERT-specific

### 4. Span masking is good prior art for the first JEPA masking policy

SpanBERT motivates masking contiguous spans instead of isolated random tokens. T5 also found span-corruption to be an effective denoising objective and used a 15% corruption rate with short spans as a strong baseline.

Implication:

- random contiguous spans are a sensible first masking policy
- an initial default corruption ratio around `0.15` is defensible
- short contiguous spans are a better starting point than fully independent token masking

## Product Decision

### Chosen v1 approach

We will implement:

- a tokenizer wrapper around `AutoTokenizer`
- a custom random block masker that samples short contiguous word spans

### Explicit v1 non-goals

We will not build these yet:

- syntax-aware masking
- semantic masking
- named-entity masking
- curriculum masking
- learned masking
- multiple masking policies in the same training run
- tokenizer training from raw text

## Architecture Decisions

## 1. Tokenizer wrapper

### Why wrap instead of calling `AutoTokenizer` directly everywhere

We need a thin stable interface for:

- YAML-driven initialization
- mask-token handling
- encode/decode normalization
- batch tokenization defaults
- future reproducibility checks

Without a wrapper, tokenizer behavior leaks into datasets, collators, maskers, and tests.

### Required wrapper responsibilities

The tokenizer component should:

- load a tokenizer from a configured Hugging Face model id
- prefer fast tokenizer implementations when available
- expose special token ids in a stable way
- ensure a usable mask token exists
- tokenize raw text into fixed-length training examples
- support returning:
  - `input_ids`
  - `attention_mask`
  - offset mappings when needed by the masker
- save tokenizer metadata needed for reproducibility

### Important mask-token caveat

Qwen3 is a causal LM family, so we should not assume a native `[MASK]` token exists with the semantics we want.

Design requirement:

- on initialization, the wrapper must verify whether `tokenizer.mask_token` is present
- if absent, the wrapper should add a dedicated mask token as a special token
- the added token string should be explicit and configurable

Recommended default:

- `"[MASK]"`

Reason:

- it is easy to read in debugging output
- it matches the current architecture docs in this repo
- it makes the masking behavior obvious in tests

Open implementation consequence:

- if the tokenizer vocabulary changes because we add a mask token, later model code must account for the new vocab size when creating embeddings

## 2. Random block masker

### Core behavior

The masker should:

- operate on tokenized text examples
- choose contiguous spans corresponding to short word blocks
- replace the chosen token positions in the context input with the configured mask token id
- preserve the original full input as the teacher-side target source

### Why word blocks instead of raw token blocks

The user requirement is phrased as masking "two words" rather than arbitrary token pieces.

With subword tokenizers, a single word can map to multiple token ids. So the masking logic should sample spans in word space first, then expand them to token positions.

This produces cleaner semantics:

- fewer partial-word masks
- easier debugging
- better alignment with the intended "block masking" concept

### Recommended v1 span policy

Use simple random contiguous word spans with:

- `min_span_words: 1`
- `max_span_words: 2`
- `mask_ratio: 0.15`
- non-overlapping spans
- no masking of special tokens
- no masking of padding

This is intentionally small and conservative.

Why this default:

- simple enough to reason about
- aligned with the requested "two words and randomized"
- close to common denoising corruption rates
- easy to test thoroughly

### Stopping rule

Sample non-overlapping spans until the number of masked non-special tokens reaches or slightly exceeds the desired target count derived from `mask_ratio`.

Important detail:

- the contract should target an approximate ratio, not exact equality
- because span boundaries operate in word space, a small tolerance is expected

## Data Contracts

## Tokenizer contract

Given raw text input, the tokenizer wrapper should produce:

- `input_ids: LongTensor (L,)`
- `attention_mask: BoolTensor or LongTensor (L,)`
- optional `offset_mapping: (L, 2)` for raw-text alignment
- tokenizer metadata:
  - `pad_token_id`
  - `mask_token_id`
  - `bos_token_id` if present
  - `eos_token_id` if present

For batched inputs:

- `input_ids: (B, L)`
- `attention_mask: (B, L)`

## Masker contract

Given a tokenized example, the masker should produce:

- `input_ids_full: (L,)`
- `input_ids_ctx: (L,)`
- `attention_mask: (L,)`
- `target_mask: BoolTensor (L,)`
- `target_positions: LongTensor (T,)`
- `target_token_ids: LongTensor (T,)`
- `masked_span_ranges_word: list[tuple[int, int]]`
- `masked_span_ranges_token: list[tuple[int, int]]`

For a padded batch, a collator can later lift this to:

- `input_ids_full: (B, L)`
- `input_ids_ctx: (B, L)`
- `attention_mask: (B, L)`
- `target_mask: (B, L)`
- `target_positions: (B, T_max)`
- `target_valid_mask: (B, T_max)`

## Word-to-token Alignment Strategy

This is the most important design detail for the masker.

### Recommended implementation order

1. Prefer fast-tokenizer alignment metadata if available.
2. If the tokenizer cannot directly provide stable word alignment for a raw string, compute word spans from raw text and project them onto token offsets using `offset_mapping`.
3. If alignment is unreliable for a sample, fail loudly in strict mode or skip the sample in exploratory mode.

### Why not just mask random token ids

Because random token masking causes three problems:

- it often masks fragments of words
- it makes debugging harder
- it drifts from the requested "two words" behavior

### Scope constraint for v1

The first implementation should optimize for English-like whitespace-separated text in tests and smoke runs.

We should keep the abstraction open for future multilingual segmentation improvements, but we should not block v1 waiting for perfect segmentation across all scripts.

## YAML Configuration Plan

Add tokenizer and masking sections to the text config.

Recommended schema:

```yaml
tokenizer:
  model_name: "Qwen/Qwen3-0.6B"
  use_fast: true
  max_length: 512
  padding: "max_length"
  truncation: true
  add_special_tokens: true
  added_mask_token: "[MASK]"
  trust_remote_code: false

masking:
  strategy: "random_word_blocks"
  mask_ratio: 0.15
  min_span_words: 1
  max_span_words: 2
  allow_overlap: false
  protect_special_tokens: true
  protect_padding: true
  seed: 42
```

### Notes on config choices

`model_name`

- lets us swap tokenizer families without code edits

`use_fast: true`

- needed for better alignment and offset metadata when available

`added_mask_token`

- keeps the mask-token policy explicit instead of hidden in code

`mask_ratio: 0.15`

- strong baseline borrowed from standard denoising practice

`min_span_words` and `max_span_words`

- directly encode the current "one to two word random block" requirement

## Recommended Module Boundaries

This PRD does not force the final package path, but the interfaces should separate responsibilities clearly.

Suggested logical modules:

- `tokenization.py`
  - load tokenizer
  - validate special tokens
  - tokenize raw text
  - expose metadata

- `masking.py`
  - build word spans
  - sample non-overlapping blocks
  - create masked context ids
  - create target-position outputs

- `test_tokenization.py`
  - tokenizer wrapper behavior

- `test_masking.py`
  - span sampling and masking behavior

## Detailed Test Plan

These tests should be treated as mandatory before downstream model integration.

## A. Tokenizer tests

### A1. Loads configured tokenizer

Verify that the wrapper reads the configured model id and initializes a tokenizer instance.

Assertions:

- model id is stored
- tokenizer object exists
- vocab size is positive

### A2. Ensures mask token exists

Verify that initialization leaves the wrapper with a valid `mask_token` and `mask_token_id`.

Assertions:

- `mask_token` is not `None`
- `mask_token_id` is not `None`
- if a new token was added, the wrapper reports that fact explicitly

### A3. Stable batch tokenization contract

Tokenize a small batch of strings.

Assertions:

- `input_ids` shape is `(B, L)`
- `attention_mask` shape is `(B, L)`
- truncation and padding obey config

### A4. Special tokens are discoverable

Assertions:

- pad token id is available
- mask token id is available
- special token ids are integers

### A5. Reproducible serialization metadata

If the wrapper saves config metadata, verify round-trip persistence of:

- model id
- added special token choice
- max length

## B. Masker tests

### B1. No special-token masking

Input a sequence containing BOS, EOS, PAD, or other special tokens.

Assertions:

- `target_mask` is false at those positions
- context ids at those positions remain unchanged

### B2. No padding masking

Assertions:

- padding positions are never selected

### B3. Contiguous span guarantee

For every sampled token span:

- token positions are contiguous
- word spans are contiguous

### B4. Non-overlap guarantee

Assertions:

- sampled spans do not overlap
- `target_positions` are unique

### B5. Approximate mask ratio

For medium-length examples:

- the masked-token fraction should be close to configured `mask_ratio`

Use a tolerance, for example:

- absolute error less than `0.05`

The exact tolerance can be tuned after implementation.

### B6. Determinism with fixed seed

Assertions:

- repeated runs with the same seed produce identical masks
- different seeds usually produce different masks

### B7. Context-target consistency

Assertions:

- `input_ids_full` preserves original token ids
- `input_ids_ctx` differs from `input_ids_full` exactly at `target_mask == True`
- those differing positions equal `mask_token_id`
- `target_token_ids` equal the original full-input token ids at `target_positions`

### B8. Very short input behavior

Test sequences too short to hit the requested ratio cleanly.

Assertions:

- function does not crash
- at least one span is chosen when masking is possible and policy says so
- no impossible positions are selected

### B9. Punctuation and repeated whitespace

Assertions:

- alignment logic remains valid
- span boundaries do not become corrupt when token offsets include punctuation

### B10. Multi-subword word behavior

Use words that split into multiple tokenizer pieces.

Assertions:

- masking a chosen word masks all token pieces belonging to that word

## C. Integration smoke tests

### C1. Tokenize then mask single example

Assertions:

- output tensors have coherent shapes
- target positions correspond to masked context positions

### C2. Tokenize then mask batch

Assertions:

- batching preserves per-sample correctness
- target-position extraction can be padded cleanly for later model use

## Research-Driven Risks

## 1. Missing native mask token in Qwen tokenizer

Risk:

- causal-model tokenizers do not always define a mask token

Mitigation:

- make mask-token presence an explicit initialization check
- add a configurable special token when absent
- record whether vocab expansion happened

## 2. Whole-word masking assumptions do not transfer cleanly to Qwen

Risk:

- off-the-shelf whole-word collators are BERT-oriented

Mitigation:

- keep masking logic custom and tokenizer-aware

## 3. Word segmentation is ambiguous across languages

Risk:

- "two words" is not universal across multilingual text

Mitigation:

- define v1 scope as English-like word segmentation
- isolate alignment logic behind a dedicated helper
- document multilingual follow-up work

## 4. Span masking may overshoot exact ratio targets

Risk:

- contiguous span sampling in word space will not always hit exact token-level ratios

Mitigation:

- define the ratio contract as approximate
- test within tolerance

## Milestone Plan

## Milestone 1: PRD and config contract

Deliverables:

- this PRD
- agreed YAML schema

Exit criteria:

- tokenizer model source and masking defaults are agreed

## Milestone 2: Tokenizer wrapper

Deliverables:

- tokenizer wrapper implementation
- tokenizer config loading
- tokenizer tests

Exit criteria:

- raw text can be tokenized reproducibly from YAML config
- mask-token handling is explicit and tested

## Milestone 3: Random block masker

Deliverables:

- word-alignment helper
- span sampler
- masking outputs
- masker tests

Exit criteria:

- masked context and target outputs satisfy the contracts above

## Milestone 4: Integration smoke tests

Deliverables:

- tokenize-plus-mask pipeline tests
- one tiny synthetic batch fixture

Exit criteria:

- downstream model code can consume these outputs without redefining data semantics

## Recommended Acceptance Criteria

We should consider the tokenizer and masker ready for downstream JEPA work only if all of the following are true:

- tokenizer model id comes from YAML
- `Qwen/Qwen3-0.6B` loads through `AutoTokenizer`
- mask token behavior is explicit and tested
- masking policy is configurable in YAML
- masking uses contiguous word blocks, not isolated token ids
- special tokens and padding are never masked
- target positions match the masked positions exactly
- tests cover short inputs, multi-subword words, and determinism

## Open Questions

These do not block the PRD, but they should be resolved before implementation starts:

1. Should v1 hard-require `use_fast=True`, or should there be a slow-tokenizer fallback path?
2. If Qwen3 lacks a native mask token, do we want to standardize on `"[MASK]"` permanently?
3. Should the first implementation live inside the legacy `jepa/` package or in a fresh text-JEPA package?
4. Should we guarantee at least one masked span per sample, even for extremely short examples?

## Recommendation

The best next implementation step is:

1. add the YAML schema for `tokenizer` and `masking`
2. build the tokenizer wrapper first
3. make mask-token handling explicit
4. build the custom word-block masker second
5. lock the behavior down with tests before touching model code

## Sources

Primary online references used for this plan:

- Hugging Face AutoTokenizer docs:
  <https://huggingface.co/docs/transformers/en/model_doc/auto>
- Hugging Face data collator docs:
  <https://huggingface.co/docs/transformers/main/en/main_classes/data_collator>
- Qwen3 model card:
  <https://huggingface.co/Qwen/Qwen3-0.6B>
- SpanBERT paper:
  <https://arxiv.org/abs/1907.10529>
- T5 paper:
  <https://arxiv.org/abs/1910.10683>
