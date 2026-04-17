from pathlib import Path

from autoresearch.train import RESULTS_COLUMNS, main


def test_validate_only_plans_run_and_initializes_results_tsv(tmp_path: Path):
    parameter_golf_root = tmp_path / "parameter-golf"
    dataset_root = parameter_golf_root / "data" / "datasets" / "fineweb10B_sp1024"
    tokenizer_path = parameter_golf_root / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    manifest_path = parameter_golf_root / "data" / "manifest.json"
    results_path = tmp_path / "results.tsv"

    dataset_root.mkdir(parents=True)
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_bytes(b"fake-tokenizer")
    (dataset_root / "fineweb_train_000000.bin").write_bytes(b"train")
    (dataset_root / "fineweb_val_000000.bin").write_bytes(b"val")
    manifest_path.write_text(
        (
            '{"datasets":[{"name":"fineweb10B_sp1024","tokenizer_name":"sp_bpe_1024",'
            '"path":"datasets/fineweb10B_sp1024","vocab_size":1024}],'
            '"tokenizers":[{"name":"sp_bpe_1024","model_path":"tokenizers/fineweb_1024_bpe.model"}]}'
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--validate-only",
            "--profile",
            "smoke",
            "--parameter-golf-root",
            str(parameter_golf_root),
            "--results-path",
            str(results_path),
            "--run-name",
            "validate-only",
        ]
    )

    assert result["mode"] == "validate"
    assert result["train_shards_available"] == 1
    assert result["val_shards_available"] == 1
    assert results_path.read_text(encoding="utf-8").splitlines()[0].split("\t") == RESULTS_COLUMNS
    assert result["trainer_argv"][0:2] == ["--config", str(Path("intertwined_hjepa.yaml").resolve())]
    assert "--parameter-golf-root" in result["trainer_argv"]
    assert "--max-steps" in result["trainer_argv"]
