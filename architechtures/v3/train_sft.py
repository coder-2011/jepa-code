from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100
GSM_ANSWER_PREFIX = "#### "


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PyTorch SFT trainer.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-gsm8k-samples", type=int, default=64)
    parser.add_argument("--gsm8k-split", default="test", choices=["train", "test"])
    parser.add_argument("--gsm8k-subset", default="main", choices=["main", "socratic"])
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output-mode",
        default="solution_only",
        choices=["solution_only", "thought_then_solution"],
    )
    parser.add_argument(
        "--reasoning-filter",
        default="any",
        choices=["any", "on", "off"],
    )
    parser.add_argument("--category", action="append", dest="categories", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_reasoning_flag(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"on", "true", "yes", "1"}:
        return "on"
    if text in {"off", "false", "no", "0", ""}:
        return "off"
    return text


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_chat_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [], ""

    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = message.get("content")
        text = normalize_text(content) if isinstance(content, str) else ""
        if not role or not text:
            continue
        normalized_messages.append({"role": role, "content": text})

    assistant_index = None
    for index in range(len(normalized_messages) - 1, -1, -1):
        if normalized_messages[index]["role"] == "assistant":
            assistant_index = index
            break

    if assistant_index is None:
        return [], ""

    return normalized_messages[:assistant_index], normalized_messages[assistant_index]["content"]


def extract_chat_supervised_pair(row: dict[str, Any]) -> tuple[str, str]:
    prompt_messages, assistant_text = split_chat_messages(row)
    user_text = ""
    for message in prompt_messages:
        if message["role"] == "user":
            user_text = message["content"]
            break
    return user_text, assistant_text


def build_assistant_target(row: dict[str, Any], output_mode: str) -> str:
    solution = normalize_text(row.get("qwen3-solution"))
    reasoning = normalize_text(row.get("qwen3-reasoning"))
    reasoning_flag = normalize_reasoning_flag(row.get("reasoning", "on"))

    if not solution and "messages" in row:
        _, assistant_text = extract_chat_supervised_pair(row)
        return assistant_text

    if output_mode == "solution_only" or not reasoning or reasoning_flag == "off":
        return solution

    return f"<thought>\n{reasoning}\n</thought>\n{solution}"


def should_keep_row(
    row: dict[str, Any], reasoning_filter: str, categories: set[str] | None
) -> bool:
    if categories is not None and normalize_text(row.get("category")) not in categories:
        return False
    if reasoning_filter == "any":
        return True
    return normalize_reasoning_flag(row.get("reasoning", "on")) == reasoning_filter


def apply_chat_template_or_fallback(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            output = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(output, dict) or hasattr(output, "keys"):
                output = output["input_ids"]
            if hasattr(output, "tolist"):
                output = output.tolist()
            if output and isinstance(output[0], list):
                output = output[0]
            return list(output)
        except Exception:
            pass

    pieces = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        pieces.append("Assistant:")
    return list(tokenizer.encode("\n".join(pieces), add_special_tokens=False))


def render_plain_supervised_messages(
    tokenizer: Any, prompt_messages: list[dict[str, str]], assistant_text: str
) -> tuple[list[int], list[int]]:
    prompt_lines = [f"{message['role'].capitalize()}: {message['content']}" for message in prompt_messages]
    prompt_lines.append("Assistant:")
    prompt_text = "\n".join(prompt_lines)
    full_text = f"{prompt_text} {assistant_text}"
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full_ids = list(tokenizer.encode(full_text, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        full_ids.append(int(tokenizer.eos_token_id))
    return prompt_ids, full_ids


def render_supervised_messages(
    tokenizer: Any, prompt_messages: list[dict[str, str]], assistant_text: str
) -> tuple[list[int], list[int]]:
    prompt_ids = apply_chat_template_or_fallback(tokenizer, prompt_messages, True)
    full_ids = apply_chat_template_or_fallback(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": assistant_text}],
        False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        prompt_ids, full_ids = render_plain_supervised_messages(
            tokenizer, prompt_messages, assistant_text
        )
    return prompt_ids, full_ids


def tokenize_supervised_example(
    tokenizer: Any, row: dict[str, Any], max_length: int, output_mode: str
) -> dict[str, list[int]] | None:
    prompt_messages = None
    problem = normalize_text(row.get("problem"))
    if not problem and "messages" in row:
        prompt_messages, _ = split_chat_messages(row)
        problem, _ = extract_chat_supervised_pair(row)

    assistant_text = build_assistant_target(row, output_mode)
    if not problem or not assistant_text:
        return None

    if prompt_messages is None:
        prompt_messages = [{"role": "user", "content": problem}]

    prompt_ids, full_ids = render_supervised_messages(tokenizer, prompt_messages, assistant_text)
    if len(prompt_ids) >= max_length:
        return None
    if len(full_ids) > max_length:
        return None

    labels = full_ids.copy()
    for idx in range(len(prompt_ids)):
        labels[idx] = IGNORE_INDEX
    return {"input_ids": full_ids, "labels": labels}


class NemotronSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        rows: list[dict[str, Any]],
        max_length: int,
        output_mode: str,
        reasoning_filter: str = "any",
        categories: set[str] | None = None,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        self.skipped_rows = 0
        for row in rows:
            if not should_keep_row(row, reasoning_filter, categories):
                continue
            tokenized = tokenize_supervised_example(
                tokenizer, row, max_length, output_mode
            )
            if tokenized is None:
                self.skipped_rows += 1
                continue
            self.examples.append(tokenized)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(example["input_ids"]) for example in examples)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for example in examples:
            pad = max_len - len(example["input_ids"])
            input_ids.append(
                example["input_ids"] + [self.pad_token_id] * pad
            )
            attention_mask.append([1] * len(example["input_ids"]) + [0] * pad)
            labels.append(example["labels"] + [IGNORE_INDEX] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def resolve_pad_token_id(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    return int(tokenizer.pad_token_id)


def resolve_torch_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_attn_implementation(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "flash_attention_2" if device == "cuda" else "sdpa"


def build_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.dtype]:
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: str | None = None):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    finally:
        importlib.util.find_spec = original_find_spec

    torch_dtype = resolve_torch_dtype(args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch_dtype,
        "attn_implementation": resolve_attn_implementation(
            args.device, args.attn_implementation
        ),
    }

    def load_model(load_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        try:
            return AutoModelForCausalLM.from_pretrained(
                args.model_name, **load_kwargs
            ), load_kwargs
        except TypeError as exc:
            if "dtype" in str(exc) and "unexpected keyword argument" in str(exc):
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                return AutoModelForCausalLM.from_pretrained(
                    args.model_name, **fallback_kwargs
                ), fallback_kwargs
            raise

    try:
        model, model_kwargs = load_model(model_kwargs)
    except Exception:
        if model_kwargs["attn_implementation"] == "flash_attention_2":
            print("flash_attention_2 load failed, retrying with sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            model, model_kwargs = load_model(model_kwargs)
        else:
            raise

    model.config.pad_token_id = resolve_pad_token_id(tokenizer)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, tokenizer, torch_dtype


def lr_multiplier(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(warmup_steps)


def should_step_optimizer(batch_index: int, num_batches: int, grad_accum_steps: int) -> bool:
    if grad_accum_steps <= 1:
        return True
    is_accum_boundary = (batch_index + 1) % grad_accum_steps == 0
    is_last_batch = batch_index + 1 == num_batches
    return is_accum_boundary or is_last_batch


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def run_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    scaler: torch.cuda.amp.GradScaler | None,
    autocast_cm: Any,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    with autocast_cm:
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(outputs.loss.detach().item())


def extract_gsm8k_answer(text: str) -> str | None:
    marker = text.rfind(GSM_ANSWER_PREFIX)
    if marker >= 0:
        candidate = text[marker + len(GSM_ANSWER_PREFIX) :].strip().splitlines()[0]
    else:
        candidate = ""
        for token in reversed(text.replace(",", "").split()):
            token = token.strip().rstrip(".")
            if token and any(char.isdigit() for char in token):
                candidate = token
                break
    cleaned = candidate.replace(",", "").strip().rstrip(".")
    return cleaned or None


def prepare_gsm8k_rows(
    split: str, subset: str, max_samples: int, seed: int
) -> list[dict[str, str]]:
    if max_samples == 0:
        return []
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=seed)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dict(row) for row in dataset]


def generate_completion(
    model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int
) -> str:
    prompt_ids = apply_chat_template_or_fallback(
        tokenizer, [{"role": "user", "content": prompt}], True
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        generated[0, input_ids.shape[1] :], skip_special_tokens=True
    )


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    device: str,
    max_new_tokens: int,
) -> float:
    if not rows:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for row in rows:
            prediction = generate_completion(
                model,
                tokenizer,
                row["question"],
                device,
                max_new_tokens,
            )
            correct += int(
                extract_gsm8k_answer(prediction) == extract_gsm8k_answer(row["answer"])
            )
    model.train()
    return correct / len(rows)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    best_gsm8k_acc: float,
) -> Path:
    checkpoint_dir = Path(output_dir) / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_saved = True
    model_save_error = None
    try:
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    except Exception as exc:
        model_saved = False
        model_save_error = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: failed to save model/tokenizer checkpoint: {exc}")

    state = {
        "step": step,
        "best_gsm8k_acc": best_gsm8k_acc,
        "args": vars(args),
    }
    if args.save_optimizer_state:
        state["optimizer"] = optimizer.state_dict()

    tmp_path = checkpoint_dir / "trainer_state.pt.tmp"
    final_path = checkpoint_dir / "trainer_state.pt"
    try:
        torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, final_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        fallback = {
            "step": step,
            "best_gsm8k_acc": best_gsm8k_acc,
            "args": vars(args),
            "model_saved": model_saved,
            "model_save_error": model_save_error,
            "optimizer_saved": False,
            "save_error": f"{type(exc).__name__}: {exc}",
        }
        with (checkpoint_dir / "trainer_state_fallback.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(fallback, handle, indent=2)
        print(f"WARNING: failed to save trainer_state.pt: {exc}")

    if model_save_error is not None and not (checkpoint_dir / "trainer_state_fallback.json").exists():
        fallback = {
            "step": step,
            "best_gsm8k_acc": best_gsm8k_acc,
            "args": vars(args),
            "model_saved": False,
            "model_save_error": model_save_error,
            "optimizer_saved": final_path.exists(),
        }
        with (checkpoint_dir / "trainer_state_fallback.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(fallback, handle, indent=2)

    latest_dir = Path(output_dir) / "latest"
    if latest_dir.exists() and latest_dir.is_symlink():
        latest_dir.unlink()
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.symlink_to(checkpoint_dir.name)
    return checkpoint_dir


def build_dataloader(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[DataLoader, NemotronSFTDataset]:
    rows = read_jsonl(args.train_file, args.max_train_samples)
    categories = set(args.categories) if args.categories else None
    dataset = NemotronSFTDataset(
        tokenizer,
        rows,
        args.max_length,
        args.output_mode,
        args.reasoning_filter,
        categories,
    )
    if len(dataset) == 0:
        raise ValueError("No training examples survived filtering/tokenization")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTCollator(tokenizer.pad_token_id),
    )
    return loader, dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, torch_dtype = build_model_and_tokenizer(args)
    model = model.to(args.device)
    train_dataloader, train_dataset = build_dataloader(args, tokenizer)
    gsm8k_rows = prepare_gsm8k_rows(
        args.gsm8k_split,
        args.gsm8k_subset,
        args.max_gsm8k_samples,
        args.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.device == "cuda" and torch_dtype == torch.float16
        else None
    )

    print(f"Loaded {len(train_dataset)} train examples (skipped {train_dataset.skipped_rows}).")
    print(f"Loaded {len(gsm8k_rows)} GSM8K validation examples.")
    print(f"Training {args.model_name} on {args.device} with dtype={torch_dtype}.")

    step = 0
    best_gsm8k_acc = -1.0
    running_loss = 0.0
    running_loss_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        num_batches = len(train_dataloader)
        for batch_index, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, args.device)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_multiplier(step, args.warmup_steps)

            if args.device == "cuda" and torch_dtype in {torch.float16, torch.bfloat16}:
                autocast_cm = torch.autocast(device_type="cuda", dtype=torch_dtype)
            else:
                autocast_cm = nullcontext()

            with autocast_cm:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(outputs.loss.detach().item())
            running_loss_steps += 1

            if not should_step_optimizer(
                batch_index, num_batches, args.grad_accum_steps
            ):
                continue

            if args.max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / float(max(running_loss_steps, 1))
                running_loss = 0.0
                running_loss_steps = 0
                print(
                    f"step={step} epoch={epoch + 1} loss={avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if gsm8k_rows and args.eval_every > 0 and step % args.eval_every == 0:
                gsm8k_acc = evaluate_gsm8k(
                    model,
                    tokenizer,
                    gsm8k_rows,
                    args.device,
                    args.gsm8k_max_new_tokens,
                )
                best_gsm8k_acc = max(best_gsm8k_acc, gsm8k_acc)
                print(f"step={step} gsm8k_acc={gsm8k_acc:.4f} best={best_gsm8k_acc:.4f}")

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_dir = save_checkpoint(
                    output_dir,
                    step,
                    model,
                    tokenizer,
                    optimizer,
                    args,
                    best_gsm8k_acc,
                )
                print(f"Saved checkpoint to {checkpoint_dir}")

    if gsm8k_rows:
        final_gsm8k_acc = evaluate_gsm8k(
            model,
            tokenizer,
            gsm8k_rows,
            args.device,
            args.gsm8k_max_new_tokens,
        )
        best_gsm8k_acc = max(best_gsm8k_acc, final_gsm8k_acc)
        print(
            f"final_step={step} final_gsm8k_acc={final_gsm8k_acc:.4f} "
            f"best_gsm8k_acc={best_gsm8k_acc:.4f}"
        )
    else:
        print(
            f"final_step={step} final_gsm8k_acc=skipped "
            f"best_gsm8k_acc={best_gsm8k_acc:.4f}"
        )

    checkpoint_dir = save_checkpoint(
        output_dir,
        step,
        model,
        tokenizer,
        optimizer,
        args,
        best_gsm8k_acc,
    )
    print(f"Saved final checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
