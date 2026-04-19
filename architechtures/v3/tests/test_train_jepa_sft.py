from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_jepa_sft", ROOT / "train_jepa_sft.py")
train_jepa_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_jepa_sft
SPEC.loader.exec_module(train_jepa_sft)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        tokens = []
        for message in messages:
            tokens.extend([len(message["role"]), len(message["content"])])
        return {"input_ids": tokens}


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()
        self.embed = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return type("Output", (), {"last_hidden_state": x})()


class TrainJEPASFTTests(unittest.TestCase):
    def test_make_example_uses_k_answer_tokens(self) -> None:
        example = train_jepa_sft.make_example(
            FakeTokenizer(),
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            max_length=16,
            k=1,
        )

        self.assertEqual(example["prompt_input_ids"], [4, 2])
        self.assertEqual(example["target_input_ids"], [4, 2, 9])
        self.assertEqual(example["target_answer_mask"], [False, False, True])
        self.assertEqual(example["labels"], [-100, -100, 9, 5])

    def test_selected_layers_defaults_to_every_other_upper_half(self) -> None:
        self.assertEqual(train_jepa_sft.selected_layers(num_layers=24, start=None, stride=2), [12, 14, 16, 18, 20, 22])

    def test_jepa_gradient_is_layer_local(self) -> None:
        torch.manual_seed(0)
        model = TinyModel()
        target = train_jepa_sft.PartialFFNTarget(model, [1], ema_decay=0.99)
        predictors = nn.ModuleDict({"1": nn.Identity()})
        batch = {
            "prompt_input_ids": torch.tensor([[1, 2, 0]]),
            "prompt_attention_mask": torch.tensor([[1, 1, 0]]),
            "target_input_ids": torch.tensor([[1, 2, 3]]),
            "target_attention_mask": torch.tensor([[1, 1, 1]]),
            "target_answer_mask": torch.tensor([[False, False, True]]),
        }

        loss = train_jepa_sft.compute_jepa_loss(model, target, predictors, batch)
        loss.backward()

        self.assertIsNone(model.embed.weight.grad)
        self.assertIsNone(model.model.layers[0].mlp.weight.grad)
        self.assertIsNotNone(model.model.layers[1].mlp.weight.grad)
        self.assertGreater(model.model.layers[1].mlp.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
