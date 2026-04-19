from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", TRAIN_PATH)
train_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_sft
SPEC.loader.exec_module(train_sft)


def load_fixture_rows() -> list[dict]:
    fixture_path = ROOT / "tests" / "fixtures" / "nemotron_tiny.jsonl"
    rows = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


class TrainSFTTests(unittest.TestCase):
    def test_first_text_field_supports_hf_aliases(self) -> None:
        row = {
            "question": "What is 1 + 1?",
            "qwen3_solution": "#### 2",
        }

        self.assertEqual(train_sft.first_text_field(row, train_sft.PROMPT_FIELDS), "What is 1 + 1?")
        self.assertEqual(train_sft.first_text_field(row, train_sft.SOLUTION_FIELDS), "#### 2")

    def test_build_assistant_target_respects_reasoning_mode(self) -> None:
        rows = load_fixture_rows()
        with_reasoning = train_sft.build_assistant_target(rows[0], "thought_then_solution")
        without_reasoning = train_sft.build_assistant_target(rows[1], "thought_then_solution")

        self.assertIn("<thought>", with_reasoning)
        self.assertTrue(with_reasoning.endswith("#### 4"))
        self.assertNotIn("<thought>", without_reasoning)
        self.assertEqual(without_reasoning, "Hello, nice to meet you.")

    def test_should_keep_row_filters_reasoning_and_category(self) -> None:
        row = load_fixture_rows()[0]

        self.assertTrue(train_sft.should_keep_row(row, "on", {"math"}))
        self.assertFalse(train_sft.should_keep_row(row, "off", {"math"}))
        self.assertFalse(train_sft.should_keep_row(row, "on", {"chat"}))

    def test_row_to_prompt_completion_uses_trl_prompt_completion_format(self) -> None:
        row = load_fixture_rows()[0]

        record = train_sft.row_to_prompt_completion(row, "solution_only")

        self.assertEqual(
            record,
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "completion": [{"role": "assistant", "content": "#### 4"}],
            },
        )

    def test_row_to_prompt_completion_requires_prompt_and_completion(self) -> None:
        self.assertIsNone(train_sft.row_to_prompt_completion({"messages": []}, "solution_only"))

    def test_build_sft_records_loads_filtered_jsonl(self) -> None:
        args = argparse.Namespace(
            train_file=str(ROOT / "tests" / "fixtures" / "nemotron_tiny.jsonl"),
            max_train_samples=None,
            reasoning_filter="on",
            categories=["math"],
            output_mode="solution_only",
        )

        records = train_sft.build_sft_records(args)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prompt"][0]["content"], "What is 2 + 2?")
        self.assertEqual(records[0]["completion"][0]["content"], "#### 4")


if __name__ == "__main__":
    unittest.main()
