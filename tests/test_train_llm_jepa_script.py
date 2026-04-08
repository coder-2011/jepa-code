from pathlib import Path
import importlib.util
import random
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from conftest import write_test_config
from text_jepa.models.llm_jepa import LLMJEPAModel


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LLM_JEPA_PATH = ROOT / "scripts" / "train_llm_jepa.py"
SPEC = importlib.util.spec_from_file_location("train_llm_jepa_script", TRAIN_LLM_JEPA_PATH)
train_llm_jepa_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_llm_jepa_script)


class FakeChatTokenizer:
    def __init__(self):
        self.chat_template = "fake"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }
        self.all_special_ids = [self.pad_token_id, self.eos_token_id]

    def add_special_tokens(self, mapping):
        for token in mapping.get("additional_special_tokens", []):
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.all_special_ids.append(token_id)

    def _tokenize_text(self, text):
        for token in sorted(self.vocab, key=len, reverse=True):
            if token.startswith("<|predictor_"):
                text = text.replace(token, f" {token} ")
        token_ids = []
        for piece in text.split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            token_ids.append(self.vocab[piece])
        return token_ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        pieces = []
        for message in messages:
            pieces.append(f"{message['role']}: {message['content']}")
            pieces.append(self.eos_token)
        if add_generation_prompt:
            pieces.append("assistant:")
        if not tokenize:
            return " ".join(pieces)
        return self._tokenize_text(" ".join(pieces))

    def decode(self, ids, skip_special_tokens=False):
        inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        tokens = [inverse_vocab[token_id] for token_id in ids]
        if skip_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in {self.pad_token, self.eos_token} and not token.startswith("<|predictor_")
            ]
        return " ".join(tokens)

    def __len__(self):
        return max(len(self.vocab), 128)


class DummyWandbRun:
    def __init__(self):
        self.logged = []
        self.alerts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log(self, data, step=None):
        self.logged.append((data, step))

    def alert(self, **kwargs):
        self.alerts.append(kwargs)


def make_fake_backbone(tokenizer):
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=max(len(tokenizer), 64),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )


def install_training_patches(monkeypatch, config_path):
    tokenizer = FakeChatTokenizer()
    run = DummyWandbRun()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name, trust_remote_code=True, torch_dtype=None):
            del model_name, trust_remote_code, torch_dtype
            return make_fake_backbone(tokenizer)

    monkeypatch.setattr(train_llm_jepa_script, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
    monkeypatch.setattr(train_llm_jepa_script, "load_tokenizer_from_yaml", lambda _path: tokenizer)
    monkeypatch.setattr(train_llm_jepa_script.wandb, "init", lambda **kwargs: run)
    return tokenizer, run, config_path


def test_train_llm_jepa_main_smoke_runs_with_additive_mask(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, model_name="hf-internal-testing/tiny-random-gpt2", max_length=64)
    train_file = ROOT / "tests" / "fixtures" / "llm_jepa_smoke_train.jsonl"
    eval_file = ROOT / "tests" / "fixtures" / "llm_jepa_smoke_eval.jsonl"
    _tokenizer, run, config_path = install_training_patches(monkeypatch, config_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_llm_jepa.py",
            "--config",
            str(config_path),
            "--train-file",
            str(train_file),
            "--eval-file",
            str(eval_file),
            "--model-name",
            "hf-internal-testing/tiny-random-gpt2",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--max-length",
            "64",
            "--device",
            "cpu",
            "--wandb-mode",
            "offline",
            "--num-workers",
            "0",
            "--save-every",
            "0",
            "--no-save-final",
            "--student-packing",
            "additive-mask",
        ],
    )

    train_llm_jepa_script.main()

    assert run.logged
    assert any("train/loss" in payload for payload, _step in run.logged)


def test_train_llm_jepa_main_smoke_runs_with_stp_random_span(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, model_name="hf-internal-testing/tiny-random-gpt2", max_length=64)
    train_file = ROOT / "tests" / "fixtures" / "llm_jepa_smoke_train.jsonl"
    eval_file = ROOT / "tests" / "fixtures" / "llm_jepa_smoke_eval.jsonl"
    _tokenizer, run, config_path = install_training_patches(monkeypatch, config_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_llm_jepa.py",
            "--config",
            str(config_path),
            "--train-file",
            str(train_file),
            "--eval-file",
            str(eval_file),
            "--model-name",
            "hf-internal-testing/tiny-random-gpt2",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--max-length",
            "64",
            "--device",
            "cpu",
            "--wandb-mode",
            "offline",
            "--num-workers",
            "0",
            "--save-every",
            "0",
            "--no-save-final",
            "--objective-mode",
            "stp-random-span",
            "--student-packing",
            "additive-mask",
            "--predictors",
            "1",
        ],
    )

    train_llm_jepa_script.main()
    captured = capsys.readouterr()

    assert "forcing --predictors 0" in captured.out
    assert run.logged
    assert any("train/jepa_loss" in payload for payload, _step in run.logged)


def test_train_llm_jepa_checkpoint_roundtrip_restores_stp_rng_state(tmp_path):
    tokenizer = FakeChatTokenizer()
    model = LLMJEPAModel(
        make_fake_backbone(tokenizer),
        objective_mode="stp_random_span",
        student_packing="additive-mask",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    expected_rng = random.Random()
    expected_rng.setstate(model._stp_rng.getstate())

    assert model._stp_rng.random() == expected_rng.random()
    train_llm_jepa_script.save_checkpoint(
        tmp_path,
        2,
        train_llm_jepa_script.checkpoint_state(2, model, optimizer, {"steps": 2}),
    )

    restored = LLMJEPAModel(
        make_fake_backbone(tokenizer),
        objective_mode="stp_random_span",
        student_packing="additive-mask",
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_step = train_llm_jepa_script.load_checkpoint(
        tmp_path / "latest.pt",
        restored,
        restored_optimizer,
        "cpu",
    )

    assert restored_step == 2
    assert restored._stp_rng.random() == expected_rng.random()
