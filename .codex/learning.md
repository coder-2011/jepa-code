# Learning Log

- For the text-JEPA tokenizer/masking milestone, treat `Qwen/Qwen3-0.6B` as a Hugging Face `AutoTokenizer` source, but do not assume it has a native JEPA-ready mask token; the wrapper should verify `mask_token` and add one explicitly if absent.
- Do not reuse Hugging Face `DataCollatorForWholeWordMask` for the Qwen path; it is BERT-oriented, so span/block masking should stay in a custom tokenizer-aware module.
- The tokenizer/masker path depends on fast-tokenizer `offset_mapping`; fail early if the loaded Hugging Face tokenizer is not fast.
- Keep masker unit tests offline and deterministic with a tiny fake tokenizer; reserve real Hugging Face loading for the tokenizer wrapper boundary.
- Layer is the codename for the text-first JEPA latent-prediction system in this workspace; use the JEPA paper as the core architectural reference and treat LLM-JEPA as adjacent precedent rather than the baseline design.
- Keep the v1 encoder compact by wrapping `nn.TransformerEncoder` with bidirectional self-attention and `(B, L)` key-padding-mask support; keep EMA as a separate utility rather than an encoder concern.
- Prefer RMSNorm over LayerNorm for the Layer encoder stack; when using `nn.TransformerEncoderLayer`, replace the built-in norms with `nn.RMSNorm` and keep the final encoder norm consistent.
- Keep EMA compact as a single `update_ema(target_module, source_module, momentum)` function; `momentum=0.0` acts as the initialization copy and ongoing updates should read `model.ema_momentum` from YAML.
- For tiny FineWeb experiments, use a local JSONL sample and filter by `token_count` / `language_score` before training; the dataset path should mask deterministically per example via `Random(seed + index)` so dataloader shuffling does not change the masking contract.
- Keep the Layer batch boundary aligned with the collated dataset output: `LayerModel.forward` should accept the full batch dict shape, including currently unused `target_mask` and `target_token_ids`, so `model(**batch)` works in the trainer without special filtering.
- Keep checkpointing in `scripts/train.py` as small script-level helpers; save step-specific files plus `latest.pt`, and note that tokenizer startup can still fail independently on Hugging Face network timeouts before any checkpoint code runs.
- The real LLM-JEPA path is a separate paired-view causal-LM setup: use `messages`/`text`/`code` JSONL examples, append predictor special tokens to the source view, compare source and target hidden states with a JEPA loss, and keep the standard LM loss in the same objective instead of reusing the Layer EMA masked-span stack.
- On Apple Silicon in this repo, prefer `mps` over `cpu` for local trainer defaults; `cuda`-only fallback logic makes LLM-JEPA look hung because a single Qwen step on CPU is extremely slow.
- For large LLM-JEPA checkpoints on a low-free-space machine, do not duplicate the payload as both `step-*.pt` and a second physical `latest.pt`; hard-link `latest.pt` to the step checkpoint, and keep the local LLM-JEPA checkpoint payload model-only to stay within disk limits.
