# Learning Log

- For the text-JEPA tokenizer/masking milestone, treat `Qwen/Qwen3-0.6B` as a Hugging Face `AutoTokenizer` source, but do not assume it has a native JEPA-ready mask token; the wrapper should verify `mask_token` and add one explicitly if absent.
- Do not reuse Hugging Face `DataCollatorForWholeWordMask` for the Qwen path; it is BERT-oriented, so span/block masking should stay in a custom tokenizer-aware module.
