from scripts.train_qwen_sft import IGNORE_INDEX, TokenizedSFTExample, collate_examples, tokenize_messages


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True):
        assert tokenize is True
        assert add_generation_prompt is False
        tokens = []
        for message in messages:
            role_id = 100 if message["role"] == "user" else 200 if message["role"] == "assistant" else 50
            content_ids = [ord(char) for char in message["content"]]
            tokens.extend([role_id, *content_ids])
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def test_tokenize_messages_masks_prompt_prefix():
    tokenizer = FakeTokenizer()
    example = tokenize_messages(
        tokenizer,
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "ab"},
            {"role": "assistant", "content": "cd"},
        ],
        max_length=32,
    )

    assert example is not None
    prompt_len = len(tokenizer.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "ab"}])["input_ids"])
    assert example.labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert example.labels[prompt_len:] == example.input_ids[prompt_len:]


def test_collate_examples_pads_labels_with_ignore_index():
    batch = [
        TokenizedSFTExample(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], labels=[IGNORE_INDEX, 2, 3]),
        TokenizedSFTExample(input_ids=[4, 5], attention_mask=[1, 1], labels=[IGNORE_INDEX, 5]),
    ]

    collated = collate_examples(batch, pad_token_id=0)

    assert tuple(collated["input_ids"].shape) == (2, 3)
    assert tuple(collated["attention_mask"].shape) == (2, 3)
    assert tuple(collated["labels"].shape) == (2, 3)
    assert collated["input_ids"].tolist()[1] == [4, 5, 0]
    assert collated["attention_mask"].tolist()[1] == [1, 1, 0]
    assert collated["labels"].tolist()[1] == [IGNORE_INDEX, 5, IGNORE_INDEX]
