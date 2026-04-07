import json
import random
from pathlib import Path

import torch

from .batching import collate_masked_examples
from .masking import mask_text_from_yaml


class FineWebJsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        config_path,
        seed=0,
        min_token_count=0,
        min_language_score=None,
        max_docs=None,
    ):
        self.tokenizer = tokenizer
        self.config_path = config_path
        self.seed = seed
        self.texts = []

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                token_count = row.get("token_count")
                # Missing token_count is only tolerated when the caller does not request a lower bound.
                if isinstance(token_count, int) and token_count < min_token_count:
                    continue
                if min_token_count > 0 and not isinstance(token_count, int):
                    continue

                language_score = row.get("language_score")
                if min_language_score is not None:
                    # Keep filtering strict so low-confidence language rows never silently pass through.
                    if not isinstance(language_score, (int, float)) or language_score < min_language_score:
                        continue

                self.texts.append(text)
                if max_docs is not None and len(self.texts) >= max_docs:
                    break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Seed per example so dataloader shuffling does not change which spans get masked.
        rng = random.Random(self.seed + index)
        return mask_text_from_yaml(
            self.tokenizer,
            self.texts[index],
            self.config_path,
            rng=rng,
        )


def create_fineweb_dataloader(
    jsonl_path,
    tokenizer,
    config_path,
    batch_size,
    shuffle=False,
    seed=0,
    num_workers=0,
    min_token_count=0,
    min_language_score=None,
    max_docs=None,
    drop_last=False,
):
    # Dataset construction owns text filtering; DataLoader owns shuffling and batch assembly.
    dataset = FineWebJsonlDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        config_path=config_path,
        seed=seed,
        min_token_count=min_token_count,
        min_language_score=min_language_score,
        max_docs=max_docs,
    )
    generator = None
    if shuffle:
        # Seed the loader generator so document order changes are reproducible across runs.
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_masked_examples,
        generator=generator,
    )


def predictor_tokens(predictors):
    return [f"<|predictor_{index}|>" for index in range(1, predictors + 1)]


def ensure_predictor_tokens(tokenizer, predictors):
    tokens = predictor_tokens(predictors)
    if tokens and hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokens


def _normalize_token_ids(token_ids):
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if hasattr(token_ids, "encodings") and token_ids.encodings:
        return list(token_ids.encodings[0].ids)
    if hasattr(token_ids, "ids"):
        return list(token_ids.ids)
    if isinstance(token_ids, dict):
        return _normalize_token_ids(token_ids["input_ids"])
    if isinstance(token_ids, list):
        if token_ids and hasattr(token_ids[0], "ids"):
            return list(token_ids[0].ids)
        return token_ids
    return token_ids


def _render_messages(tokenizer, messages, add_generation_prompt=False):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return _normalize_token_ids(token_ids)

    text = []
    for message in messages:
        text.append(f"{message['role']}: {message['content']}")
    if add_generation_prompt:
        text.append("assistant:")
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode("\n".join(text), add_special_tokens=True)

    encoded = tokenizer(
        "\n".join(text),
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return _normalize_token_ids(encoded["input_ids"])


def _pad_ids(token_ids, max_length, pad_token_id, *, field_name, allow_truncation):
    token_ids = list(token_ids)
    original_length = len(token_ids)
    if original_length > max_length:
        if not allow_truncation:
            raise ValueError(
                f"{field_name} length {original_length} exceeds max_length={max_length}; "
                "refusing to truncate because it would silently change the JEPA view contract"
            )
        token_ids = token_ids[:max_length]

    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_length:
        token_ids.append(pad_token_id)
        attention_mask.append(0)
    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _last_content_index(token_ids, eos_token_id, max_length):
    if not token_ids:
        return 0
    index = min(len(token_ids), max_length) - 1
    if token_ids[index] == eos_token_id and index > 0:
        index -= 1
    return index


def _extract_views(row):
    messages = row["messages"]
    text_messages = row.get("text")
    code_messages = row.get("code")
    if text_messages is None or code_messages is None:
        if not messages:
            raise ValueError("paired LLM-JEPA examples require non-empty messages")
        if messages[-1]["role"] != "assistant":
            raise ValueError("paired LLM-JEPA message-only rows must end with an assistant turn")
        prefix_messages = messages[:-1]
        if not prefix_messages:
            raise ValueError("paired LLM-JEPA message-only rows require source context before the assistant turn")
        if any(message["role"] == "assistant" for message in prefix_messages):
            raise ValueError(
                "paired LLM-JEPA message-only rows must be single-turn prompt->assistant examples; "
                "provide explicit `text` and `code` views for multi-turn conversations"
            )
        text_messages = prefix_messages
        code_messages = [messages[-1]]
    if not text_messages or not code_messages:
        raise ValueError("paired LLM-JEPA examples require user/text and assistant/code views")
    return messages, text_messages, code_messages


class LLMJEPAPairedJsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length,
        predictors=1,
        max_docs=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.predictors = predictors
        self.rows = []
        self.predictor_token_strings = ensure_predictor_tokens(tokenizer, predictors)

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "messages" not in row:
                    continue
                self.rows.append(row)
                if max_docs is not None and len(self.rows) >= max_docs:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        messages, text_messages, code_messages = _extract_views(self.rows[index])
        source_messages = json.loads(json.dumps(text_messages))
        if self.predictor_token_strings:
            source_messages[-1]["content"] += "".join(self.predictor_token_strings)

        full_token_ids = _render_messages(self.tokenizer, messages, add_generation_prompt=False)
        prompt_token_ids = _render_messages(self.tokenizer, messages[:-1], add_generation_prompt=True)
        source_token_ids = _render_messages(self.tokenizer, source_messages, add_generation_prompt=False)
        target_token_ids = _render_messages(self.tokenizer, code_messages, add_generation_prompt=False)

        input_ids, attention_mask = _pad_ids(
            full_token_ids,
            self.max_length,
            self.tokenizer.pad_token_id,
            field_name="messages",
            allow_truncation=True,
        )
        labels = input_ids.clone()
        labels[: min(len(prompt_token_ids), self.max_length)] = -100
        labels[attention_mask == 0] = -100

        source_input_ids, source_attention_mask = _pad_ids(
            source_token_ids,
            self.max_length,
            self.tokenizer.pad_token_id,
            field_name="text view",
            allow_truncation=False,
        )
        target_input_ids, target_attention_mask = _pad_ids(
            target_token_ids,
            self.max_length,
            self.tokenizer.pad_token_id,
            field_name="code view",
            allow_truncation=False,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_input_ids": source_input_ids,
            "source_attention_mask": source_attention_mask,
            "source_last_index": torch.tensor(
                _last_content_index(source_token_ids, self.tokenizer.eos_token_id, self.max_length),
                dtype=torch.long,
            ),
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask,
            "target_last_index": torch.tensor(
                _last_content_index(target_token_ids, self.tokenizer.eos_token_id, self.max_length),
                dtype=torch.long,
            ),
        }


def create_llm_jepa_dataloader(
    jsonl_path,
    tokenizer,
    max_length,
    batch_size,
    predictors=1,
    shuffle=False,
    seed=0,
    num_workers=0,
    max_docs=None,
    drop_last=False,
):
    dataset = LLMJEPAPairedJsonlDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_length=max_length,
        predictors=predictors,
        max_docs=max_docs,
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
    )
