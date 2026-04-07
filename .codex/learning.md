# Learning Log

- For the text-JEPA tokenizer/masking milestone, treat `Qwen/Qwen3-0.6B` as a Hugging Face `AutoTokenizer` source, but do not assume it has a native JEPA-ready mask token; the wrapper should verify `mask_token` and add one explicitly if absent.
- Do not reuse Hugging Face `DataCollatorForWholeWordMask` for the Qwen path; it is BERT-oriented, so span/block masking should stay in a custom tokenizer-aware module.
- The tokenizer/masker path depends on fast-tokenizer `offset_mapping`; fail early if the loaded Hugging Face tokenizer is not fast.
- Keep masker unit tests offline and deterministic with a tiny fake tokenizer; reserve real Hugging Face loading for the tokenizer wrapper boundary.
