from pathlib import Path

import torch

from autoresearch.train import RESULTS_COLUMNS, build_sentencepiece_luts, main


class _FakeSentencePiece:
    def vocab_size(self) -> int:
        return 6

    def is_control(self, token_id: int) -> bool:
        return token_id == 0

    def is_unknown(self, token_id: int) -> bool:
        return token_id == 1

    def is_unused(self, token_id: int) -> bool:
        return token_id == 2

    def is_byte(self, token_id: int) -> bool:
        return token_id == 5

    def id_to_piece(self, token_id: int) -> str:
        pieces = {
            3: "foo",
            4: "▁bar",
            5: "<0x61>",
        }
        return pieces[token_id]


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


def test_build_sentencepiece_luts_matches_parameter_golf_semantics():
    base_bytes, has_leading_space, is_boundary = build_sentencepiece_luts(
        _FakeSentencePiece(),
        vocab_size=8,
        device=torch.device("cpu"),
    )

    assert base_bytes.shape[0] == 8
    assert has_leading_space.shape[0] == 8
    assert is_boundary.shape[0] == 8

    # Control / unknown / unused ids are excluded from byte counting.
    assert base_bytes[0].item() == 0
    assert base_bytes[1].item() == 0
    assert base_bytes[2].item() == 0
    assert is_boundary[0].item() is True
    assert is_boundary[1].item() is True
    assert is_boundary[2].item() is True

    # Normal text pieces count UTF-8 bytes; leading-space pieces defer the space to context logic.
    assert base_bytes[3].item() == len("foo".encode("utf-8"))
    assert base_bytes[4].item() == len("bar".encode("utf-8"))
    assert has_leading_space[3].item() is False
    assert has_leading_space[4].item() is True
    assert is_boundary[3].item() is False
    assert is_boundary[4].item() is False

    # Byte fallback pieces count as one byte.
    assert base_bytes[5].item() == 1
    assert is_boundary[5].item() is False
