from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train_qwen_sft import build_dataset, collate_examples, parse_args as parse_train_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate masked SFT loss and optional greedy generations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split", default="chat")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--show-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if bool(args.train_file) == bool(args.dataset_name):
        parser.error("Provide exactly one of --train-file or --dataset-name.")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return getattr(torch, requested)


def train_args_from_eval(args: argparse.Namespace) -> argparse.Namespace:
    return parse_train_args(
        [
            "--train-file" if args.train_file else "--dataset-name",
            args.train_file or args.dataset_name,
            "--output-dir",
            "tmp-eval",
            "--model-name",
            args.model,
            "--max-length",
            str(args.max_length),
            "--dataset-split",
            args.dataset_split,
        ]
        + (["--max-train-samples", str(args.max_train_samples)] if args.max_train_samples is not None else [])
    )


def load_messages(path: str, limit: int | None) -> list[list[dict[str, Any]]]:
    rows = []
    for i, line in enumerate(Path(path).open("r", encoding="utf-8")):
        if limit is not None and i >= limit:
            break
        row = json.loads(line)
        messages = row.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            rows.append(messages)
    return rows


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)
    is_peft_adapter = model_path.is_dir() and (model_path / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        token=args.hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_args = train_args_from_eval(args)
    dataset, skipped, dropped = build_dataset(tokenizer, train_args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )

    if is_peft_adapter:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype if device.type != "cpu" else torch.float32,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            token=args.hf_token,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype if device.type != "cpu" else torch.float32,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            token=args.hf_token,
        ).to(device)
    model.eval()

    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            outputs = model(**batch)
            masked_tokens = int(batch["labels"].ne(-100).sum().item())
            loss_sum += float(outputs.loss.item()) * masked_tokens
            token_count += masked_tokens

    result = {
        "model": args.model,
        "is_peft_adapter": is_peft_adapter,
        "examples": len(dataset),
        "skipped": skipped,
        "dropped": dropped,
        "masked_tokens": token_count,
        "avg_masked_loss": loss_sum / token_count,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }

    if args.show_samples and args.train_file:
        sample_messages = load_messages(args.train_file, args.show_samples)
        generations = []
        with torch.inference_mode():
            for messages in sample_messages:
                prompt_messages = messages[:-1]
                reference = str(messages[-1].get("content", "")).strip()
                inputs = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                generations.append(
                    {
                        "prompt_roles": [str(message.get("role", "")) for message in prompt_messages],
                        "reference": reference,
                        "completion": completion,
                    }
                )
        result["samples"] = generations

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
