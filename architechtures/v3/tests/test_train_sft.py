from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_sft", ROOT / "train_sft.py")
train_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_sft
SPEC.loader.exec_module(train_sft)


def load_fixture_rows() -> list[dict]:
    with (ROOT / "tests" / "fixtures" / "nemotron_tiny.jsonl").open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class TrainSFTTests(unittest.TestCase):
    def test_row_messages_uses_chat_rows_directly(self) -> None:
        messages = train_sft.row_messages(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
            "solution_only",
        )
        self.assertEqual(messages[-1], {"role": "assistant", "content": "Hello"})

    def test_row_messages_builds_fallback_prompt_completion(self) -> None:
        messages = train_sft.row_messages(load_fixture_rows()[0], "thought_then_solution")
        self.assertEqual(messages[0], {"role": "user", "content": "What is 2 + 2?"})
        self.assertIn("<thought>", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("#### 4"))

    def test_tokenize_messages_masks_prompt_prefix(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
                assert tokenize is True
                assert add_generation_prompt is False
                assert return_dict is True
                tokens = []
                for message in messages:
                    tokens.extend([len(message["role"]), len(message["content"])])
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        tokenized = train_sft.tokenize_messages(
            FakeTokenizer(),
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            max_length=32,
        )
        self.assertEqual(tokenized["labels"], [-100, -100, -100, -100, 9, 5])

    def test_build_dataset_filters_and_drops(self) -> None:
        rows = [
            {
                "category": "chat",
                "reasoning": "off",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            {
                "category": "chat",
                "reasoning": "on",
                "messages": [
                    {"role": "user", "content": "A" * 50},
                    {"role": "assistant", "content": "B" * 50},
                ],
            },
        ]
        args = argparse.Namespace(
            train_file="unused",
            dataset_name=None,
            dataset_split="chat",
            max_train_samples=None,
            categories=["chat"],
            reasoning_filter="any",
            output_mode="solution_only",
            max_length=10,
        )
        original = train_sft.load_rows
        train_sft.load_rows = lambda _: rows
        try:
            dataset, skipped, dropped = train_sft.build_dataset(
                tokenizer=type(
                    "FakeTokenizer",
                    (),
                    {
                        "apply_chat_template": staticmethod(
                            lambda messages, tokenize, add_generation_prompt, return_dict: {
                                "input_ids": list(range(sum(len(m["content"]) for m in messages))),
                                "attention_mask": [1] * sum(len(m["content"]) for m in messages),
                            }
                        )
                    },
                )(),
                args=args,
            )
        finally:
            train_sft.load_rows = original
        self.assertEqual(len(dataset), 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(dropped, 1)

    def test_load_nvidia_rows_respects_max_rows(self) -> None:
        calls = []
        import sys as _sys
        import types as _types

        fake_hf = _types.ModuleType("huggingface_hub")
        fake_hf.list_repo_files = lambda *args, **kwargs: ["data/chat-00000-of-00002.parquet", "data/chat-00001-of-00002.parquet"]
        fake_hf.hf_hub_download = lambda **kwargs: calls.append(kwargs["filename"]) or kwargs["filename"]
        fake_pyarrow = _types.ModuleType("pyarrow")
        fake_pq = _types.ModuleType("pyarrow.parquet")
        fake_pq.read_table = lambda path: _types.SimpleNamespace(
            to_pylist=lambda: [{"id": path + "-0"}, {"id": path + "-1"}]
        )
        fake_pyarrow.parquet = fake_pq
        _sys.modules["huggingface_hub"] = fake_hf
        _sys.modules["pyarrow"] = fake_pyarrow
        _sys.modules["pyarrow.parquet"] = fake_pq
        self.assertEqual(
            train_sft.load_nvidia_rows("chat", 3),
            [
                {"id": "data/chat-00000-of-00002.parquet-0"},
                {"id": "data/chat-00000-of-00002.parquet-1"},
                {"id": "data/chat-00001-of-00002.parquet-0"},
            ],
        )
        self.assertEqual(calls, ["data/chat-00000-of-00002.parquet", "data/chat-00001-of-00002.parquet"])


if __name__ == "__main__":
    unittest.main()
