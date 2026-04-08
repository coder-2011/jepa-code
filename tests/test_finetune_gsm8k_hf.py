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

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_kwargs):
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


class FakeQwenThinkingTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        pieces = []
        for message in messages:
            if message["role"] == "assistant":
                pieces.append(f"assistant: <think> </think> {message['content']}")
            else:
                pieces.append(f"{message['role']}: {message['content']}")
            pieces.append(self.eos_token)
        if add_generation_prompt:
            pieces.append("assistant:")
            if kwargs.get("enable_thinking") is False:
                pieces.append("<think> </think>")
        rendered = " ".join(pieces)
        if not tokenize:
            return rendered
        return self._tokenize_text(rendered)


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


def test_gsm8k_sft_dataset_builds_causal_lm_labels(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    dataset = finetune_script.GSM8KSFTJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=64,
    )

    example = dataset[0]

    assert set(example) == {"input_ids", "attention_mask", "labels"}
    assert example["input_ids"].shape == (64,)
    assert example["attention_mask"].shape == (64,)
    assert example["labels"].shape == (64,)
    assert (example["labels"] == -100).any()
    assert (example["labels"] != -100).any()


def test_gsm8k_sft_dataset_masks_qwen_no_think_scaffold(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    tokenizer = FakeQwenThinkingTokenizer()
    dataset = finetune_script.GSM8KSFTJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=64,
    )

    example = dataset[0]
    label_positions = (example["labels"] != -100).nonzero(as_tuple=False).flatten()
    first_label = int(label_positions[0])
    supervised_text = tokenizer.decode(example["input_ids"][first_label : first_label + 4].tolist())

    assert supervised_text.startswith("1 + 1 =")
    assert "<think>" not in supervised_text


def test_gsm8k_sft_dataset_skips_truncated_rows_without_assistant_targets(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    tokenizer = FakeChatTokenizer()
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    full_token_ids = finetune_script.render_token_ids(
        tokenizer,
        rows[0]["messages"],
        add_generation_prompt=False,
    )
    prompt_token_ids = finetune_script.prompt_token_ids_for_full_prefix(
        tokenizer,
        rows[0]["messages"],
        full_token_ids,
    )

    try:
        finetune_script.GSM8KSFTJsonlDataset(
            jsonl_path=data_path,
            tokenizer=tokenizer,
            max_length=len(prompt_token_ids),
        )
    except ValueError as exc:
        assert "No superviseable assistant targets" in str(exc)
    else:
        raise AssertionError("Expected all-prompt truncated rows to be rejected")


def test_prompt_prefix_mismatch_is_rejected():
    class BrokenTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
            token_ids = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            if add_generation_prompt:
                return [999] + token_ids
            return token_ids

    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    tokenizer = BrokenTokenizer()
    full_token_ids = finetune_script.render_token_ids(tokenizer, row["messages"], add_generation_prompt=False)

    try:
        finetune_script.prompt_token_ids_for_full_prefix(tokenizer, row["messages"], full_token_ids)
    except ValueError as exc:
        assert "not a prefix" in str(exc)
    else:
        raise AssertionError("Expected mismatched prompt/full rendering to be rejected")


def test_gsm8k_sft_dataset_respects_max_docs(tmp_path):
    data_path = tmp_path / "gsm8k.jsonl"
    write_gsm8k_rows(data_path)
    rows_text = data_path.read_text(encoding="utf-8")
    with data_path.open("a", encoding="utf-8") as handle:
        handle.write(rows_text)

    dataset = finetune_script.GSM8KSFTJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeChatTokenizer(),
        max_length=64,
        max_docs=1,
    )

    assert len(dataset) == 1


def test_infer_lora_target_modules_prefers_full_qkvo_projection_set():
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.k_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)
            self.o_proj = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Linear(4, 4)

    model = torch.nn.Module()
    model.attn = TinyAttention()

    assert finetune_script.infer_lora_target_modules(model) == ["q_proj", "k_proj", "v_proj", "o_proj"]
