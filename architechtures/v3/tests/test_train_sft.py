from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", TRAIN_PATH)
train_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_sft
SPEC.loader.exec_module(train_sft)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self) -> None:
        self._vocab = {"<pad>": 0, "<eos>": 1}

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.replace("\n", " \n ").split()
        token_ids = [self._token_id(token) for token in tokens]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        text = []
        for message in messages:
            text.append(f"{message['role']}: {message['content']}")
        if add_generation_prompt:
            text.append("assistant:")
        return self.encode("\n".join(text), add_special_tokens=True)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(inverse[token_id])
        return " ".join(tokens)


def load_fixture_rows() -> list[dict]:
    fixture_path = ROOT / "tests" / "fixtures" / "nemotron_tiny.jsonl"
    rows = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        del attention_mask
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=train_sft.IGNORE_INDEX,
            )
        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})()


class TrainSFTTests(unittest.TestCase):
    def test_split_chat_messages_preserves_system_prompt(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }

        prompt_messages, assistant_text = train_sft.split_chat_messages(row)

        self.assertEqual(
            prompt_messages,
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hi"},
            ],
        )
        self.assertEqual(assistant_text, "Hello")

    def test_build_assistant_target_respects_reasoning_mode(self) -> None:
        rows = load_fixture_rows()
        with_reasoning = train_sft.build_assistant_target(rows[0], "thought_then_solution")
        without_reasoning = train_sft.build_assistant_target(rows[1], "thought_then_solution")

        self.assertIn("<thought>", with_reasoning)
        self.assertTrue(with_reasoning.endswith("#### 4"))
        self.assertNotIn("<thought>", without_reasoning)
        self.assertEqual(without_reasoning, "Hello, nice to meet you.")

    def test_tokenize_supervised_example_masks_prompt_tokens(self) -> None:
        tokenizer = DummyTokenizer()
        row = load_fixture_rows()[0]

        tokenized = train_sft.tokenize_supervised_example(
            tokenizer=tokenizer,
            row=row,
            max_length=128,
            output_mode="solution_only",
        )

        self.assertIsNotNone(tokenized)
        labels = tokenized["labels"]
        self.assertEqual(labels[0], train_sft.IGNORE_INDEX)
        self.assertTrue(any(label != train_sft.IGNORE_INDEX for label in labels))

    def test_tokenize_supervised_example_drops_truncated_examples(self) -> None:
        tokenizer = DummyTokenizer()
        row = {
            "problem": "Short prompt",
            "qwen3-solution": "one two three four five six seven eight",
            "reasoning": "off",
        }

        tokenized = train_sft.tokenize_supervised_example(
            tokenizer=tokenizer,
            row=row,
            max_length=6,
            output_mode="solution_only",
        )

        self.assertIsNone(tokenized)

    def test_dataset_filters_reasoning_and_category(self) -> None:
        tokenizer = DummyTokenizer()
        rows = load_fixture_rows()
        dataset = train_sft.NemotronSFTDataset(
            tokenizer=tokenizer,
            rows=rows,
            max_length=128,
            output_mode="solution_only",
            reasoning_filter="on",
            categories={"math"},
        )

        self.assertEqual(len(dataset), 1)

    def test_extract_gsm8k_answer_supports_marker_and_fallback(self) -> None:
        self.assertEqual(train_sft.extract_gsm8k_answer("work\n#### 12"), "12")
        self.assertEqual(train_sft.extract_gsm8k_answer("The answer is 12."), "12")
        self.assertIsNone(train_sft.extract_gsm8k_answer("No numeric answer"))

    def test_should_step_optimizer_flushes_final_partial_accumulation(self) -> None:
        decisions = [
            train_sft.should_step_optimizer(batch_index=i, num_batches=7, grad_accum_steps=4)
            for i in range(7)
        ]

        self.assertEqual(decisions, [False, False, False, True, False, False, True])

    def test_run_train_step_smoke(self) -> None:
        tokenizer = DummyTokenizer()
        rows = load_fixture_rows()
        dataset = train_sft.NemotronSFTDataset(
            tokenizer=tokenizer,
            rows=rows,
            max_length=128,
            output_mode="solution_only",
        )
        batch = train_sft.SFTCollator(pad_token_id=tokenizer.pad_token_id)([dataset[0], dataset[1]])

        model = TinyCausalLM(vocab_size=256, hidden_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = train_sft.run_train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            scaler=None,
            autocast_cm=train_sft.nullcontext(),
            grad_accum_steps=1,
            max_grad_norm=1.0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
