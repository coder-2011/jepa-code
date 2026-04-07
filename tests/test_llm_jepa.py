import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from text_jepa.data import LLMJEPAPairedJsonlDataset, create_llm_jepa_dataloader
from text_jepa.models.llm_jepa import LLMJEPAModel
from text_jepa.train.llm_jepa_step import train_llm_jepa_step


class FakeChatTokenizer:
    def __init__(self):
        self.chat_template = "fake"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }

    def add_special_tokens(self, mapping):
        for token in mapping.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def _tokenize_text(self, text):
        for token in sorted(self.vocab, key=len, reverse=True):
            if token.startswith("<|predictor_"):
                text = text.replace(token, f" {token} ")
        token_ids = []
        for piece in text.split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            token_ids.append(self.vocab[piece])
        return token_ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        pieces = []
        for message in messages:
            pieces.append(f"{message['role']}: {message['content']}")
            pieces.append(self.eos_token)
        if add_generation_prompt:
            pieces.append("assistant:")
        if not tokenize:
            return " ".join(pieces)
        return self._tokenize_text(" ".join(pieces))

    def __len__(self):
        return len(self.vocab)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeEncodingTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        token_ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return token_ids
        return FakeEncoding(token_ids)


class FakeBatchEncodingTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        token_ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return token_ids
        return {"input_ids": token_ids}


def write_messages_jsonl(path):
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match dog then digit"},
                {"role": "assistant", "content": "dog[0-9]"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match cat then vowel"},
                {"role": "assistant", "content": "cat[aeiou]"},
            ]
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_llm_jepa_dataset_builds_lm_and_view_tensors(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        predictors=2,
    )

    example = dataset[0]
    predictor_ids = [
        tokenizer.vocab["<|predictor_1|>"],
        tokenizer.vocab["<|predictor_2|>"],
    ]

    assert example["input_ids"].shape == (24,)
    assert example["labels"].shape == (24,)
    assert example["source_input_ids"].shape == (24,)
    assert example["target_input_ids"].shape == (24,)
    assert example["source_last_index"].item() == int(example["source_attention_mask"].sum().item()) - 2
    assert example["source_input_ids"][example["source_last_index"]].item() == predictor_ids[-1]
    assert example["source_input_ids"][example["source_last_index"] - 1].item() == predictor_ids[0]
    assert (example["labels"] == -100).any()
    assert (example["labels"] != -100).any()


def test_llm_jepa_dataset_accepts_encoding_objects_from_chat_template(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeEncodingTokenizer()

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        predictors=1,
    )

    example = dataset[0]

    assert example["input_ids"].shape == (24,)
    assert example["source_input_ids"].shape == (24,)
    assert example["target_input_ids"].shape == (24,)


def test_llm_jepa_dataset_accepts_batch_encoding_like_chat_template(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeBatchEncodingTokenizer()

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        predictors=1,
    )

    example = dataset[0]

    assert example["input_ids"].shape == (24,)
    assert example["source_input_ids"].shape == (24,)
    assert example["target_input_ids"].shape == (24,)


def test_llm_jepa_model_combines_lm_and_jepa_losses(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    dataloader = create_llm_jepa_dataloader(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        batch_size=2,
        predictors=1,
    )
    batch = next(iter(dataloader))

    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=24,
            n_ctx=24,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model = LLMJEPAModel(backbone, lambda_jepa=0.5, gamma_lm=1.0, ema_momentum=0.9)

    outputs = model(**batch)

    assert outputs["loss"].ndim == 0
    assert outputs["lm_loss"].ndim == 0
    assert outputs["jepa_loss"].ndim == 0
    assert torch.allclose(outputs["loss"], outputs["lm_loss"] + 0.5 * outputs["jepa_loss"])
    assert not outputs["target_embeddings"].requires_grad


def test_llm_jepa_target_backbone_is_gradient_free(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    dataloader = create_llm_jepa_dataloader(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        batch_size=2,
        predictors=1,
    )
    batch = next(iter(dataloader))

    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=24,
            n_ctx=24,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model = LLMJEPAModel(backbone, ema_momentum=0.5)

    for parameter in model.target_backbone.parameters():
        assert not parameter.requires_grad

    outputs = model(**batch)
    outputs["loss"].backward()

    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.target_backbone.parameters())


def test_train_llm_jepa_step_updates_parameters(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    dataloader = create_llm_jepa_dataloader(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        batch_size=2,
        predictors=1,
    )
    batch = next(iter(dataloader))

    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=24,
            n_ctx=24,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model = LLMJEPAModel(backbone, ema_momentum=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.backbone.parameters()]
    target_before = [parameter.detach().clone() for parameter in model.target_backbone.parameters()]

    outputs = train_llm_jepa_step(model, optimizer, batch)

    assert outputs["loss"].ndim == 0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.backbone.parameters())
    )
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(target_before, model.target_backbone.parameters())
    )
