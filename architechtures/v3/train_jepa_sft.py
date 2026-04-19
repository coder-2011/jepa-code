from __future__ import annotations
import argparse
import copy
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_sft


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-local answer-conditioned JEPA-SFT.")
    parser.add_argument("--train-file")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split", default="chat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--output-mode", default="solution_only", choices=["solution_only", "thought_then_solution"])
    parser.add_argument("--reasoning-filter", default="any", choices=["any", "on", "off"])
    parser.add_argument("--category", action="append", dest="categories")

    parser.add_argument("--jepa-lambda", type=float, default=0.01)
    parser.add_argument("--jepa-k", type=int, default=32)
    parser.add_argument("--jepa-ema-decay", type=float, default=0.99)
    parser.add_argument("--jepa-layer-start", type=int)
    parser.add_argument("--jepa-layer-stride", type=int, default=2)
    parser.add_argument("--predictor-hidden-mult", type=float, default=1.0)
    parser.add_argument("--predictor-activation", default="gelu", choices=["gelu", "silu"])

    args = parser.parse_args()
    if bool(args.train_file) == bool(args.dataset_name):
        parser.error("Provide exactly one of --train-file or --dataset-name.")
    if args.jepa_k < 1:
        parser.error("--jepa-k must be >= 1.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(requested: str):
    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, requested)


def make_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    k: int,
) -> dict[str, list[int] | list[bool]] | None:
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=False, return_dict=True)
    full_ids = list(full["input_ids"])
    prompt_ids = list(prompt["input_ids"])
    if len(full_ids) > max_length or not input_prefix(full_ids, prompt_ids):
        return None

    answer_len = len(full_ids) - len(prompt_ids)
    if answer_len < 1:
        return None
    target_answer_len = min(k, answer_len)

    target_ids = full_ids[: len(prompt_ids) + target_answer_len]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": [1] * len(prompt_ids),
        "target_input_ids": target_ids,
        "target_attention_mask": [1] * len(target_ids),
        "target_answer_mask": [False] * len(prompt_ids) + [True] * target_answer_len,
    }


def input_prefix(full_ids: list[int], prompt_ids: list[int]) -> bool:
    return full_ids[: len(prompt_ids)] == prompt_ids


def build_examples(tokenizer: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int, int]:
    categories = set(args.categories) if args.categories else None
    examples: list[dict[str, Any]] = []
    skipped = 0
    dropped = 0

    for row in train_sft.load_rows(args):
        if categories is not None and str(row.get("category", "")).strip() not in categories:
            continue
        if args.reasoning_filter != "any" and train_sft.normalize_reasoning(row.get("reasoning", "on")) != args.reasoning_filter:
            continue
        messages = train_sft.row_messages(row, args.output_mode)
        if messages is None:
            skipped += 1
            continue
        example = make_example(tokenizer, messages, args.max_length, args.jepa_k)
        if example is None:
            dropped += 1
            continue
        examples.append(example)

    if not examples:
        raise ValueError("No examples survived filtering and max-length constraints.")
    return examples, skipped, dropped


def pad_1d(values: list[int] | list[bool], length: int, pad_value: int | bool) -> list[int] | list[bool]:
    return values + [pad_value] * (length - len(values))


def collate_examples(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_full = max(len(x["input_ids"]) for x in batch)
    max_prompt = max(len(x["prompt_input_ids"]) for x in batch)
    max_target = max(len(x["target_input_ids"]) for x in batch)

    return {
        "input_ids": torch.tensor([pad_1d(x["input_ids"], max_full, pad_token_id) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad_1d(x["attention_mask"], max_full, 0) for x in batch], dtype=torch.long),
        "labels": torch.tensor([pad_1d(x["labels"], max_full, IGNORE_INDEX) for x in batch], dtype=torch.long),
        "prompt_input_ids": torch.tensor([pad_1d(x["prompt_input_ids"], max_prompt, pad_token_id) for x in batch], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([pad_1d(x["prompt_attention_mask"], max_prompt, 0) for x in batch], dtype=torch.long),
        "target_input_ids": torch.tensor([pad_1d(x["target_input_ids"], max_target, pad_token_id) for x in batch], dtype=torch.long),
        "target_attention_mask": torch.tensor([pad_1d(x["target_attention_mask"], max_target, 0) for x in batch], dtype=torch.long),
        "target_answer_mask": torch.tensor([pad_1d(x["target_answer_mask"], max_target, False) for x in batch], dtype=torch.bool),
    }


def selected_layers(num_layers: int, start: int | None, stride: int) -> list[int]:
    start = num_layers // 2 if start is None else start
    layers = list(range(start, num_layers, stride))
    if not layers:
        raise ValueError("No JEPA layers selected.")
    return layers


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, hidden_mult: float, activation: str) -> None:
        super().__init__()
        hidden = max(hidden_size, int(hidden_size * hidden_mult))
        act = nn.GELU() if activation == "gelu" else nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            act,
            nn.Linear(hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom


class PartialFFNTarget(nn.Module):
    def __init__(self, model: nn.Module, layer_ids: list[int], ema_decay: float) -> None:
        super().__init__()
        self.model = model
        self.layer_ids = layer_ids
        self.ema_decay = ema_decay
        self.target_ffns = nn.ModuleDict(
            {str(i): copy.deepcopy(self.layers[i].mlp).requires_grad_(False) for i in layer_ids}
        )

    @property
    def layers(self):
        return self.model.model.layers

    @contextmanager
    def _swapped_ffns(self):
        original = {i: self.layers[i].mlp for i in self.layer_ids}
        try:
            for i in self.layer_ids:
                self.layers[i].mlp = self.target_ffns[str(i)]
            yield
        finally:
            for i, module in original.items():
                self.layers[i].mlp = module

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        hooks = [
            self.target_ffns[str(i)].register_forward_hook(
                lambda _module, _args, output, layer_id=i: outputs.__setitem__(layer_id, output.detach())
            )
            for i in self.layer_ids
        ]
        try:
            with self._swapped_ffns():
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

    @torch.no_grad()
    def ema_update(self) -> None:
        for i in self.layer_ids:
            target = self.target_ffns[str(i)]
            source = self.layers[i].mlp
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(self.ema_decay).add_(source_param, alpha=1.0 - self.ema_decay)


@contextmanager
def capture_mlp_inputs(model: nn.Module, layer_ids: list[int]):
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        model.model.layers[i].mlp.register_forward_pre_hook(
            lambda _module, args, layer_id=i: captured.__setitem__(layer_id, args[0].detach())
        )
        for i in layer_ids
    ]
    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_jepa_loss(
    model: nn.Module,
    target: PartialFFNTarget,
    predictors: nn.ModuleDict,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer_ids = target.layer_ids
    with torch.no_grad(), capture_mlp_inputs(model, layer_ids) as student_inputs:
        model(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            use_cache=False,
        )
        target_outputs = target(batch["target_input_ids"], batch["target_attention_mask"])

    prompt_mask = batch["prompt_attention_mask"].bool()
    target_mask = batch["target_answer_mask"]
    losses = []
    for layer_id in layer_ids:
        student_local = model.model.layers[layer_id].mlp(student_inputs[layer_id])
        student_repr = masked_mean(student_local, prompt_mask)
        target_repr = masked_mean(target_outputs[layer_id], target_mask).detach()
        prediction = predictors[str(layer_id)](student_repr)
        losses.append(F.mse_loss(prediction.float(), target_repr.float()))
    return torch.stack(losses).mean()


def save_jepa_state(
    output_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    target: PartialFFNTarget,
    predictors: nn.ModuleDict,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    torch.save(
        {
            "target_ffns": target.target_ffns.state_dict(),
            "predictors": predictors.state_dict(),
            "args": vars(args),
            "layer_ids": target.layer_ids,
        },
        output_dir / "jepa_state.pt",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples, skipped, dropped = build_examples(tokenizer, args)
    print(f"Loaded {len(examples)} JEPA-SFT examples (skipped {skipped}, dropped {dropped}).")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=resolve_torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.train()
    model.config.use_cache = False

    layer_ids = selected_layers(len(model.model.layers), args.jepa_layer_start, args.jepa_layer_stride)
    print(f"JEPA layers: {layer_ids}")
    target = PartialFFNTarget(model, layer_ids, args.jepa_ema_decay).to(device)
    predictors = nn.ModuleDict(
        {
            str(i): Predictor(model.config.hidden_size, args.predictor_hidden_mult, args.predictor_activation)
            for i in layer_ids
        }
    ).to(device=device, dtype=next(model.parameters()).dtype)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictors.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            micro_step += 1
            batch = move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            lm_loss = outputs.loss
            jepa_loss = compute_jepa_loss(model, target, predictors, batch)
            loss = lm_loss + args.jepa_lambda * jepa_loss
            (loss / args.grad_accum_steps).backward()

            should_step = micro_step % args.grad_accum_steps == 0
            is_last = micro_step == len(dataloader) * args.epochs
            if should_step or is_last:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(predictors.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                target.ema_update()
                global_step += 1

                if global_step % args.logging_steps == 0 or global_step == 1:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "lm_loss": round(lm_loss.item(), 4),
                                "jepa_loss": round(jepa_loss.item(), 4),
                                "loss": round(loss.item(), 4),
                            }
                        )
                    )

    save_jepa_state(Path(args.output_dir), model, tokenizer, target, predictors, args)


if __name__ == "__main__":
    main()
