import json
from copy import deepcopy
import random
from unittest.mock import patch
import warnings
from types import SimpleNamespace

import torch
from torch import nn
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


class DummyBaseModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        assert "output_hidden_states" not in kwargs
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.base_model = DummyBaseModel(hidden_size=hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask, labels=None, use_cache=False):
        hidden_states = self.base_model(input_ids, attention_mask, use_cache=use_cache).last_hidden_state
        loss = hidden_states.sum() * 0.0
        return SimpleNamespace(loss=loss)


def write_messages_jsonl(path):
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match dog then digit"},
                {"role": "assistant", "content": "dog[0-9]"},
            ],
            "text": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match dog then digit"},
            ],
            "code": [
                {"role": "assistant", "content": "dog[0-9]"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match cat then vowel"},
                {"role": "assistant", "content": "cat[aeiou]"},
            ],
            "text": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match cat then vowel"},
            ],
            "code": [
                {"role": "assistant", "content": "cat[aeiou]"},
            ],
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
    assert example["packed_input_ids"].shape == (24,)
    assert example["packed_attention_mask"].shape == (24,)
    assert example["packed_labels"].shape == (24,)
    assert example["packed_valid"].item() is True
    assert example["source_last_index"].item() == int(example["source_attention_mask"].sum().item()) - 2
    assert example["source_input_ids"][example["source_last_index"]].item() == predictor_ids[-1]
    assert example["source_input_ids"][example["source_last_index"] - 1].item() == predictor_ids[0]
    assert example["packed_source_length"].item() == int(example["source_attention_mask"].sum().item())
    assert example["packed_target_length"].item() == int(example["target_attention_mask"].sum().item())
    assert example["packed_source_last_index"].item() == example["source_last_index"].item()
    assert (
        example["packed_target_last_index"].item()
        == example["packed_source_length"].item() + example["target_last_index"].item()
    )
    assert example["packed_source_span_start"].item() == 0
    assert example["packed_source_span_end"].item() == example["packed_source_length"].item()
    assert example["packed_target_span_start"].item() == example["packed_source_length"].item()
    assert (
        example["packed_target_span_end"].item()
        == example["packed_source_length"].item() + example["packed_target_length"].item()
    )
    assert torch.all(example["packed_labels"] == -100)
    assert (example["labels"] == -100).any()
    assert (example["labels"] != -100).any()


def test_llm_jepa_dataset_accepts_single_turn_message_only_rows(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "match dog then digit"},
                {"role": "assistant", "content": "dog[0-9]"},
            ]
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=24,
        predictors=1,
    )

    example = dataset[0]

    assert example["input_ids"].shape == (24,)
    assert example["source_input_ids"].shape == (24,)
    assert example["target_input_ids"].shape == (24,)


def test_llm_jepa_dataset_rejects_multi_turn_message_only_rows(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a regex."},
                {"role": "assistant", "content": "dog[0-9]"},
                {"role": "user", "content": "Now explain it."},
                {"role": "assistant", "content": "It matches dog plus a digit."},
            ]
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=32,
        predictors=1,
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "single-turn prompt->assistant examples" in str(exc)
    else:
        raise AssertionError("Expected the dataset to reject ambiguous multi-turn message-only rows")


def test_llm_jepa_dataset_rejects_truncated_text_view(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "one two three four five six seven eight nine ten eleven twelve"},
                {"role": "assistant", "content": "dog[0-9]"},
            ],
            "text": [
                {"role": "system", "content": "Convert text to regex."},
                {"role": "user", "content": "one two three four five six seven eight nine ten eleven twelve"},
            ],
            "code": [
                {"role": "assistant", "content": "dog[0-9]"},
            ],
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=8,
        predictors=2,
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "text view length" in str(exc)
        assert "refusing to truncate" in str(exc)
    else:
        raise AssertionError("Expected the dataset to reject truncated text views")


def test_llm_jepa_dataset_rejects_truncated_code_view(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Convert."},
                {"role": "user", "content": "dog digit"},
                {"role": "assistant", "content": "one two three four five six seven eight nine ten eleven twelve"},
            ],
            "text": [
                {"role": "system", "content": "Convert."},
                {"role": "user", "content": "dog digit"},
            ],
            "code": [
                {"role": "assistant", "content": "one two three four five six seven eight nine ten eleven twelve"},
            ],
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=8,
        predictors=1,
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "code view length" in str(exc)
        assert "refusing to truncate" in str(exc)
    else:
        raise AssertionError("Expected the dataset to reject truncated code views")


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


def test_llm_jepa_dataset_marks_packed_rows_invalid_when_only_the_combined_tube_overflows(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "beta"},
            ],
            "text": [
                {"role": "user", "content": "alpha"},
            ],
            "code": [
                {"role": "assistant", "content": "beta"},
            ],
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=5,
        predictors=0,
    )

    example = dataset[0]

    assert example["source_attention_mask"].sum().item() == 3
    assert example["target_attention_mask"].sum().item() == 3
    assert example["packed_valid"].item() is False
    assert example["packed_attention_mask"].sum().item() == 0
    assert example["packed_source_length"].item() == 0
    assert example["packed_target_length"].item() == 0


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


def test_llm_jepa_model_reads_final_hidden_state_without_hidden_state_stack(tmp_path):
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

    model = LLMJEPAModel(DummyBackbone(hidden_size=8), ema_momentum=0.5)
    outputs = model(**batch)

    assert outputs["source_embeddings"].shape == (2, 8)
    assert outputs["target_embeddings"].shape == (2, 8)


def test_llm_jepa_model_supports_additive_mask_student_packing(tmp_path):
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
    model = LLMJEPAModel(
        backbone,
        lambda_jepa=0.5,
        gamma_lm=1.0,
        ema_momentum=0.9,
        student_packing="additive-mask",
    )

    outputs = model(**batch)

    assert outputs["loss"].ndim == 0
    assert outputs["lm_loss"].ndim == 0
    assert outputs["jepa_loss"].ndim == 0
    assert outputs["source_embeddings"].shape == (2, 16)
    assert outputs["target_embeddings"].shape == (2, 16)


def test_llm_jepa_additive_mask_matches_separate_student_path_in_eval_mode(tmp_path):
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
    separate_model = LLMJEPAModel(
        backbone,
        lambda_jepa=0.5,
        gamma_lm=1.0,
        ema_momentum=0.9,
        student_packing="separate",
    ).eval()
    additive_model = LLMJEPAModel(
        deepcopy(backbone),
        lambda_jepa=0.5,
        gamma_lm=1.0,
        ema_momentum=0.9,
        student_packing="additive-mask",
    ).eval()

    with torch.no_grad():
        separate_outputs = separate_model(**batch)
        additive_outputs = additive_model(**batch)

    assert torch.allclose(separate_outputs["lm_loss"], additive_outputs["lm_loss"])
    assert torch.allclose(separate_outputs["jepa_loss"], additive_outputs["jepa_loss"])
    assert torch.allclose(separate_outputs["source_embeddings"], additive_outputs["source_embeddings"])
    assert torch.allclose(separate_outputs["target_embeddings"], additive_outputs["target_embeddings"])


def test_llm_jepa_model_supports_stp_random_span_mode(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    dataloader = create_llm_jepa_dataloader(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        batch_size=2,
        predictors=0,
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
    model = LLMJEPAModel(
        backbone,
        lambda_jepa=0.5,
        gamma_lm=1.0,
        objective_mode="stp_random_span",
        student_packing="additive-mask",
        stp_samples=2,
        ema_momentum=0.9,
    )

    outputs = model(**batch)

    assert outputs["loss"].ndim == 0
    assert outputs["lm_loss"].ndim == 0
    assert outputs["jepa_loss"].ndim == 0
    assert outputs["stp_loss"].ndim == 0
    assert torch.allclose(outputs["jepa_loss"], outputs["stp_loss"])


def test_llm_jepa_additive_mask_mode_rejects_examples_without_valid_packed_rows(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "beta"},
            ],
            "text": [
                {"role": "user", "content": "alpha"},
            ],
            "code": [
                {"role": "assistant", "content": "beta"},
            ],
        }
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    tokenizer = FakeChatTokenizer()
    batch = next(
        iter(
            create_llm_jepa_dataloader(
                jsonl_path=data_path,
                tokenizer=tokenizer,
                max_length=5,
                batch_size=1,
                predictors=0,
            )
        )
    )
    model = LLMJEPAModel(DummyBackbone(hidden_size=8), student_packing="additive-mask")

    try:
        model(**batch)
    except ValueError as exc:
        assert "requires source+target packed rows" in str(exc)
    else:
        raise AssertionError("Expected additive-mask mode to reject batches without valid packed rows")


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


def test_train_llm_jepa_step_skips_ema_for_stp_mode(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    dataloader = create_llm_jepa_dataloader(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=24,
        batch_size=2,
        predictors=0,
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
    model = LLMJEPAModel(
        backbone,
        objective_mode="stp_random_span",
        student_packing="additive-mask",
        stp_samples=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    outputs = train_llm_jepa_step(model, optimizer, batch)

    assert outputs["loss"].ndim == 0
    assert model.target_backbone is None


def test_llm_jepa_stp_mode_passes_explicit_packed_span_bounds(tmp_path):
    data_path = tmp_path / "paired.jsonl"
    write_messages_jsonl(data_path)
    tokenizer = FakeChatTokenizer()
    batch = next(
        iter(
            create_llm_jepa_dataloader(
                jsonl_path=data_path,
                tokenizer=tokenizer,
                max_length=24,
                batch_size=2,
                predictors=0,
            )
        )
    )

    model = LLMJEPAModel(
        GPT2LMHeadModel(
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
        ),
        objective_mode="stp_random_span",
        student_packing="additive-mask",
    )

    with patch("text_jepa.models.llm_jepa.random_span_batch_loss", return_value=torch.tensor(0.25)) as mocked_loss:
        outputs = model(**batch)

    _, kwargs = mocked_loss.call_args
    assert torch.equal(kwargs["source_span_starts"], batch["packed_source_span_start"])
    assert torch.equal(kwargs["source_span_ends"], batch["packed_source_span_end"])
    assert torch.equal(kwargs["target_span_starts"], batch["packed_target_span_start"])
    assert torch.equal(kwargs["target_span_ends"], batch["packed_target_span_end"])
    assert outputs["jepa_loss"].item() == 0.25


def test_llm_jepa_state_dict_restores_stp_rng_state():
    model = LLMJEPAModel(
        DummyBackbone(hidden_size=8),
        objective_mode="stp_random_span",
        student_packing="additive-mask",
    )
    expected_rng = random.Random()
    expected_rng.setstate(model._stp_rng.getstate())

    first_draw = model._stp_rng.random()
    expected_first_draw = expected_rng.random()
    state_dict = model.state_dict()
    second_draw = expected_rng.random()
    restored = LLMJEPAModel(
        DummyBackbone(hidden_size=8),
        objective_mode="stp_random_span",
        student_packing="additive-mask",
    )
    restored.load_state_dict(state_dict)

    assert first_draw == expected_first_draw
    assert restored._stp_rng.random() == second_draw


def test_llm_jepa_warns_when_cosine_metric_is_used_with_2d_hidden_states():
    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=2,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMJEPAModel(backbone, jepa_metric="cosine")

    assert any("geometrically degenerate" in str(warning.message) for warning in caught)
