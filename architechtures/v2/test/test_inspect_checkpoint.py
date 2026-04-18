from pathlib import Path

import json
import torch

from intertwined_hjepa import IntertwinedConfig, IntertwinedHJEPA
from scripts.inspect_checkpoint import (
    load_checkpoint,
    resolve_parameter_golf_assets,
    sample_next_id,
    token_repetition_stats,
)


def test_load_checkpoint_fills_missing_sigreg_config_from_yaml(tmp_path: Path):
    config = IntertwinedConfig.from_yaml("intertwined_hjepa.yaml")
    model = IntertwinedHJEPA(config)
    old_config = {
        key: value
        for key, value in config.__dict__.items()
        if not key.startswith("sigreg_") and key != "beta_sigreg"
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"config": old_config, "model": model.state_dict()}, checkpoint_path)

    loaded_model, checkpoint, missing_fields = load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert checkpoint["config"] == old_config
    assert loaded_model.config.beta_sigreg == config.beta_sigreg
    assert loaded_model.config.sigreg_num_slices == config.sigreg_num_slices
    assert set(missing_fields) == {
        "beta_sigreg",
        "sigreg_warmup_steps",
        "sigreg_num_slices",
        "sigreg_t_max",
        "sigreg_n_points",
    }


def test_resolve_parameter_golf_assets(tmp_path: Path):
    root = tmp_path / "parameter-golf"
    dataset_root = root / "data" / "datasets" / "fineweb10B_sp1024"
    tokenizer_path = root / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    dataset_root.mkdir(parents=True)
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_bytes(b"fake")
    manifest_path = root / "data" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "fineweb10B_sp1024",
                        "tokenizer_name": "sp_bpe_1024",
                        "path": "datasets/fineweb10B_sp1024",
                        "vocab_size": 1024,
                    }
                ],
                "tokenizers": [
                    {
                        "name": "sp_bpe_1024",
                        "model_path": "tokenizers/fineweb_1024_bpe.model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assets = resolve_parameter_golf_assets(root, "sp1024")

    assert assets.dataset_root == dataset_root.resolve()
    assert assets.tokenizer_path == tokenizer_path.resolve()
    assert assets.vocab_size == 1024


def test_sample_next_id_uses_top_k_subset():
    logits = torch.tensor([0.0, 10.0, 1.0])

    assert sample_next_id(logits, temperature=1.0, top_k=1) == 1


def test_token_repetition_stats_counts_adjacent_repeats_and_runs():
    stats = token_repetition_stats([4, 4, 4, 7, 8, 8])

    assert stats == {
        "unique_new_tokens": 3,
        "repeated_fraction": 3 / 5,
        "max_run": 3,
    }
