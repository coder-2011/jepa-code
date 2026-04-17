from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset_helpers import HEADER_DTYPE, HEADER_INTS, SHARD_MAGIC, SHARD_VERSION, TOKEN_DTYPE
from scripts.train_intertwined_hjepa import main


def write_shard(path: Path, tokens: list[int], *, magic: int = SHARD_MAGIC, version: int = SHARD_VERSION):
    header = np.zeros((HEADER_INTS,), dtype=HEADER_DTYPE)
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(handle)


def test_trainer_writes_latest_checkpoint(tmp_path: Path):
    dataset_root = tmp_path / "fineweb10B_sp1024"
    write_shard(dataset_root / "fineweb_train_000000.bin", list(range(1, 33)))
    write_shard(dataset_root / "fineweb_val_000000.bin", list(range(33, 65)))

    config_path = tmp_path / "intertwined_hjepa.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 256,
                "max_length": 4,
                "residual_dim": 8,
                "compressed_dim": 4,
                "depth": 3,
                "num_heads": 2,
                "predictor_hidden_dim": 16,
                "dropout": 0.0,
                "ema_momentum": 0.5,
                "lambda_jepa": 0.1,
                "jepa_warmup_steps": 0,
                "jepa_dropout_rate": 0.0,
                "beta_sigreg": 0.0,
                "sigreg_warmup_steps": 0,
                "sigreg_num_slices": 8,
                "sigreg_t_max": 3.0,
                "sigreg_n_points": 5,
                "tie_weights": True,
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-steps",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--save-every",
            "1",
            "--run-name",
            "minimal",
            "--out-dir",
            str(tmp_path / "runs"),
            "--wandb-mode",
            "disabled",
        ]
    )

    latest_path = result["run_dir"] / "latest.pt"
    step_path = result["run_dir"] / "step-000002.pt"
    assert latest_path.exists()
    assert step_path.exists()

    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["tokens_processed"] == 16
    assert checkpoint["config"]["max_length"] == 4


def test_trainer_counts_jepa_dropout_steps(tmp_path: Path):
    dataset_root = tmp_path / "fineweb10B_sp1024"
    write_shard(dataset_root / "fineweb_train_000000.bin", list(range(1, 33)))
    write_shard(dataset_root / "fineweb_val_000000.bin", list(range(33, 65)))

    config_path = tmp_path / "intertwined_hjepa.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 256,
                "max_length": 4,
                "residual_dim": 8,
                "compressed_dim": 4,
                "depth": 3,
                "num_heads": 2,
                "predictor_hidden_dim": 16,
                "dropout": 0.0,
                "ema_momentum": 0.5,
                "lambda_jepa": 0.1,
                "jepa_warmup_steps": 0,
                "jepa_dropout_rate": 1.0,
                "beta_sigreg": 0.05,
                "sigreg_warmup_steps": 0,
                "sigreg_num_slices": 8,
                "sigreg_t_max": 3.0,
                "sigreg_n_points": 5,
                "tie_weights": True,
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-steps",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "0",
            "--save-every",
            "0",
            "--run-name",
            "dropout",
            "--out-dir",
            str(tmp_path / "runs"),
            "--wandb-mode",
            "disabled",
        ]
    )

    assert result["jepa_dropout_steps"] == 2
    assert result["jepa_dropout_fraction"] == 1.0
