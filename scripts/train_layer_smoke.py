from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
# Mirror the main training script's import behavior so the smoke check can run from the repo root.
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.models.layer_model import LayerModel
from text_jepa.train.step import train_step


def load_config():
    # Smoke training uses the default config on purpose so it exercises the same settings users start from.
    with (ROOT / "text-jepa-default.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config):
    model_config = config["model"]
    # The smoke path keeps vocab size synthetic but otherwise respects the default model configuration.
    return LayerModel(
        vocab_size=128,
        max_length=config["tokenizer"]["max_length"],
        hidden_dim=model_config["hidden_dim"],
        encoder_num_layers=model_config["num_layers"],
        encoder_num_heads=model_config["num_heads"],
        encoder_ffn_dim=model_config["ffn_dim"],
        predictor_num_layers=2,
        predictor_num_heads=model_config["num_heads"],
        predictor_ffn_dim=model_config["ffn_dim"],
        dropout=model_config["dropout"],
        ema_momentum=model_config["ema_momentum"],
    )


def make_batch():
    # This batch mirrors the minimal training contract without depending on tokenization or dataset code.
    return {
        "input_ids_full": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long),
        "input_ids_ctx": torch.tensor([[1, 9, 3, 4], [5, 6, 9, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long),
        "target_positions": torch.tensor([[1, 3], [2, 0]], dtype=torch.long),
        "target_valid_mask": torch.tensor([[True, False], [True, False]]),
    }


def main():
    config = load_config()
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One training step is enough to verify the forward, backward, optimizer, and EMA path together.
    outputs = train_step(model, optimizer, make_batch())
    print(f"smoke loss: {outputs['loss'].item():.6f}")


if __name__ == "__main__":
    main()
