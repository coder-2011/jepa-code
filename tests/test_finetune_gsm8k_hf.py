from pathlib import Path
import importlib.util
import json

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "finetune_gsm8k_hf.py"
SPEC = importlib.util.spec_from_file_location("finetune_gsm8k_hf_script", SCRIPT_PATH)
finetune_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(finetune_script)


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

    def _tokenize_text(self, text):
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
        rendered = " ".join(pieces)
        if not tokenize:
            return rendered
        return self._tokenize_text(rendered)

    def decode(self, ids, skip_special_tokens=False):
        inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        tokens = [inverse_vocab[token_id] for token_id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {self.pad_token, self.eos_token}]
        return " ".join(tokens)

    def __len__(self):
        return len(self.vocab)


def write_gsm8k_rows(path):
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Solve the problem."},
                {"role": "user", "content": "What is one plus one?"},
                {"role": "assistant", "content": "1 + 1 = 2\n#### 2"},
            ],
            "text": [
                {"role": "system", "content": "Solve the problem."},
                {"role": "user", "content": "What is one plus one?"},
            ],
            "code": [
                {"role": "assistant", "content": "1 + 1 = 2\n#### 2"},
            ],
        }
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_causal_lm_view_dataset_reuses_llm_jepa_label_contract(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    base_dataset = finetune_script.LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=64,
        predictors=0,
    )

    dataset = finetune_script.CausalLMViewDataset(base_dataset)
    example = dataset[0]

    assert set(example) == {"input_ids", "attention_mask", "labels"}
    assert example["input_ids"].shape == (64,)
    assert example["attention_mask"].shape == (64,)
    assert example["labels"].shape == (64,)
    assert (example["labels"] == -100).any()
    assert (example["labels"] != -100).any()


def test_causal_lm_view_dataset_skips_truncated_rows_without_assistant_targets(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    tokenizer = FakeChatTokenizer()
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    source_length = len(tokenizer.apply_chat_template(rows[0]["text"], tokenize=True, add_generation_prompt=False))
    base_dataset = finetune_script.LLMJEPAPairedJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=source_length,
        predictors=0,
    )

    try:
        finetune_script.CausalLMViewDataset(base_dataset)
    except ValueError as exc:
        assert "No superviseable assistant targets" in str(exc)
    else:
        raise AssertionError("Expected all-prompt truncated rows to be rejected")


def test_infer_lora_target_modules_prefers_q_and_v_projection_pair():
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Linear(4, 4)

    model = torch.nn.Module()
    model.attn = TinyAttention()

    assert finetune_script.infer_lora_target_modules(model) == ["q_proj", "v_proj"]
