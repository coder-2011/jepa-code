from pathlib import Path
import math

import torch
import pytest
import yaml

from autoresearch.train import RESULTS_COLUMNS, build_sentencepiece_luts, evaluate_bpb, main, parse_args as parse_autoresearch_args
from scripts.train_intertwined_hjepa import parse_args as parse_trainer_args


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
    assert results_path.read_text(encoding="utf-8").splitlines()[0].split("\t") == RESULTS_COLUMNS
    assert result["trainer_argv"][0:2] == ["--config", str(Path("intertwined_hjepa.yaml").resolve())]
    assert "--parameter-golf-root" in result["trainer_argv"]
    assert "--max-steps" in result["trainer_argv"]
    assert "--no-compile" not in result["trainer_argv"]


def test_validate_only_materializes_autoresearch_config_overrides(tmp_path: Path):
    results_path = tmp_path / "results.tsv"
    out_dir = tmp_path / "runs"

    result = main(
        [
            "--validate-only",
            "--profile",
            "smoke",
            "--parameter-golf-root",
            str(tmp_path / "parameter-golf"),
            "--results-path",
            str(results_path),
            "--out-dir",
            str(out_dir),
            "--run-name",
            "override-smoke",
            "--jepa-dropout-rate",
            "0.1",
            "--auxiliary-layer-start",
            "3",
            "--auxiliary-layer-stride",
            "2",
        ]
    )

    config_path = Path(result["trainer_argv"][result["trainer_argv"].index("--config") + 1])
    assert config_path == out_dir / "override-smoke" / "autoresearch_config.yaml"
    config_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config_values["jepa_dropout_rate"] == 0.1
    assert config_values["auxiliary_layer_start"] == 3
    assert config_values["auxiliary_layer_stride"] == 2


def test_no_compile_flag_is_not_supported():
    with pytest.raises(SystemExit):
        parse_trainer_args(["--no-compile"])
    with pytest.raises(SystemExit):
        parse_autoresearch_args(["--parameter-golf-root", "/tmp/parameter-golf", "--no-compile"])


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


class _PerfectNextTokenModel:
    class _Config:
        vocab_size = 8

    def __init__(self) -> None:
        self.config = self._Config()
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, *, input_ids, labels, **_kwargs):
        logits = torch.full((*labels.shape, self.config.vocab_size), -100.0, device=labels.device)
        logits.scatter_(2, labels.unsqueeze(-1), 100.0)
        return {"logits": logits}


def test_evaluate_bpb_uses_loader_shifted_labels_directly():
    model = _PerfectNextTokenModel()
    tokenizer = _FakeSentencePiece()
    input_ids = torch.tensor([[3, 4, 3]], dtype=torch.long)
    labels = torch.tensor([[4, 3, 5]], dtype=torch.long)

    bpb = evaluate_bpb(
        model,
        tokenizer=tokenizer,
        eval_loader=[(input_ids, labels)],
        device=torch.device("cpu"),
        step=0,
    )

    assert bpb == 0.0
    assert model.training is True


def test_evaluate_bpb_keeps_zero_byte_targets_in_loss_numerator():
    class _ModelWithControlledLogits:
        class _Config:
            vocab_size = 8

        def __init__(self) -> None:
            self.config = self._Config()
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def __call__(self, *, input_ids, labels, **_kwargs):
            logits = torch.full((*labels.shape, self.config.vocab_size), -100.0, device=labels.device)
            # Position 0 predicts the zero-byte control token poorly on purpose.
            logits[0, 0, 3] = 0.0
            logits[0, 0, 0] = -1.0
            # Position 1 predicts a normal token perfectly.
            logits[0, 1, 4] = 100.0
            return {"logits": logits}

    model = _ModelWithControlledLogits()
    tokenizer = _FakeSentencePiece()
    input_ids = torch.tensor([[3, 3]], dtype=torch.long)
    labels = torch.tensor([[0, 4]], dtype=torch.long)

    bpb = evaluate_bpb(
        model,
        tokenizer=tokenizer,
        eval_loader=[(input_ids, labels)],
        device=torch.device("cpu"),
        step=0,
    )

    logits = torch.full((2, 8), -100.0)
    logits[0, 3] = 0.0
    logits[0, 0] = -1.0
    logits[1, 4] = 100.0
    losses = torch.nn.functional.cross_entropy(logits, labels.reshape(-1), reduction="none")
    target_bytes = len("bar".encode("utf-8")) + 1  # leading-space piece after a non-boundary token
    expected = float(losses.sum().item() / (math.log(2.0) * target_bytes))

    assert bpb == pytest.approx(expected)
    assert model.training is True
