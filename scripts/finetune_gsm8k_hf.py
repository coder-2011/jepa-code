from __future__ import annotations

from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.env import load_local_env
from text_jepa.runtime import default_device, validate_device
from text_jepa.utils.repro import configure_reproducibility

load_local_env(ROOT)

from text_jepa.data import _normalize_token_ids, _pad_ids


DEFAULT_TRAIN_FILE = ROOT / "tmp" / "gsm8k" / "gsm8k_train.jsonl"
DEFAULT_EVAL_FILE = ROOT / "tmp" / "gsm8k" / "gsm8k_test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "tmp" / "runs" / "gsm8k_sft_qwen3_0_6b"


def parse_args():
    parser = ArgumentParser(description="LoRA fine-tune a causal LM on GSM8K JSONL rows using Hugging Face Trainer.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--eval-max-docs", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    parser.add_argument("--tf32", dest="tf32", action="store_true")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-final", dest="save_final", action="store_true")
    parser.add_argument("--no-save-final", dest="save_final", action="store_false")
    parser.add_argument("--merge-adapters", dest="merge_adapters", action="store_true")
    parser.add_argument("--no-merge-adapters", dest="merge_adapters", action="store_false")
    parser.add_argument("--save-adapters-final", dest="save_adapters_final", action="store_true")
    parser.add_argument("--no-save-adapters-final", dest="save_adapters_final", action="store_false")
    parser.add_argument("--resume-from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--wandb-project", default="gsm8k-sft-hf")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules")
    parser.set_defaults(
        pin_memory=None,
        grad_checkpointing=True,
        tf32=True,
        save_final=True,
        merge_adapters=True,
        save_adapters_final=False,
    )
    return parser.parse_args()


def choose_wandb_mode(requested_mode):
    if requested_mode is not None:
        return requested_mode
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def resolve_torch_dtype(dtype_name, device):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def dtype_flags(dtype_name, device):
    if dtype_name == "float16":
        return False, device == "cuda"
    if dtype_name == "bfloat16":
        return device == "cuda", False
    if dtype_name == "auto":
        return device == "cuda", False
    return False, False


def render_token_ids(tokenizer, messages, *, add_generation_prompt, enable_thinking=None):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return _normalize_token_ids(tokenizer.apply_chat_template(messages, **kwargs))

    rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    if add_generation_prompt:
        rendered += "\nassistant:"
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(rendered, add_special_tokens=True)
    encoded = tokenizer(rendered, truncation=False, padding=False, return_attention_mask=False)
    return _normalize_token_ids(encoded["input_ids"])


def prompt_token_ids_for_full_prefix(tokenizer, messages, full_token_ids):
    prefix_messages = messages[:-1]
    prompt_token_ids = render_token_ids(
        tokenizer,
        prefix_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if full_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
        return prompt_token_ids

    prompt_token_ids = render_token_ids(
        tokenizer,
        prefix_messages,
        add_generation_prompt=True,
    )
    if full_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
        return prompt_token_ids

    raise ValueError(
        "Rendered prompt is not a prefix of the rendered full conversation; "
        "refusing to build ambiguous SFT labels."
    )


class GSM8KSFTJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length, max_docs=None):
        self.examples = []
        skipped = 0
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "messages" not in row:
                    continue
                example = self._build_example(row, tokenizer, max_length)
                if example is None:
                    skipped += 1
                    continue
                self.examples.append(example)
                if max_docs is not None and len(self.examples) >= max_docs:
                    break
        if not self.examples:
            raise ValueError("No superviseable assistant targets remain after applying max_length")
        self.skipped = skipped

    @staticmethod
    def _build_example(row, tokenizer, max_length):
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("GSM8K SFT rows must contain messages ending with an assistant answer")

        full_token_ids = render_token_ids(tokenizer, messages, add_generation_prompt=False)
        prompt_token_ids = prompt_token_ids_for_full_prefix(tokenizer, messages, full_token_ids)

        input_ids, attention_mask = _pad_ids(
            full_token_ids,
            max_length,
            tokenizer.pad_token_id,
            field_name="messages",
            allow_truncation=True,
        )
        labels = input_ids.clone()
        labels[: min(len(prompt_token_ids), max_length)] = -100
        labels[attention_mask == 0] = -100
        if torch.all(labels == -100):
            return None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def infer_lora_target_modules(model):
    leaf_names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if name and hasattr(module, "weight")
    }
    preferred_groups = (
        ("q_proj", "v_proj"),
        ("q_proj", "k_proj", "v_proj", "o_proj"),
        ("c_attn", "c_proj"),
        ("query_key_value", "dense"),
    )
    for group in preferred_groups:
        found = [module_name for module_name in group if module_name in leaf_names]
        if len(found) >= 2:
            return found
    raise ValueError(
        "Could not infer LoRA target modules from the model. "
        "Pass --lora-target-modules explicitly, for example q_proj,v_proj."
    )


def load_model_and_tokenizer(args):
    torch_dtype = resolve_torch_dtype(args.dtype, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    model.config.use_cache = False

    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = (
        [module_name.strip() for module_name in args.lora_target_modules.split(",") if module_name.strip()]
        if args.lora_target_modules
        else infer_lora_target_modules(model)
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer, target_modules


def build_datasets(args, tokenizer):
    train_dataset = GSM8KSFTJsonlDataset(
        jsonl_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_docs=args.max_docs,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No rows loaded from {args.train_file}")
    eval_dataset = None
    if args.eval_file:
        eval_dataset = GSM8KSFTJsonlDataset(
            jsonl_path=args.eval_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_docs=args.eval_max_docs,
        )
        if len(eval_dataset) == 0:
            raise ValueError(f"No rows loaded from {args.eval_file}")
    return train_dataset, eval_dataset


def training_arguments(args, *, report_to):
    bf16, fp16 = dtype_flags(args.dtype, args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused" if args.device == "cuda" else "adamw_torch",
        bf16=bf16,
        fp16=fp16,
        tf32=args.tf32 if args.device == "cuda" else None,
        gradient_checkpointing=args.grad_checkpointing,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps" if args.eval_every > 0 else "no",
        eval_steps=args.eval_every if args.eval_every > 0 else None,
        save_strategy="steps" if args.save_every > 0 else "no",
        save_steps=args.save_every if args.save_every > 0 else 500,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        report_to=report_to,
        run_name=args.wandb_name,
        prediction_loss_only=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        dataloader_persistent_workers=bool(args.num_workers > 0),
        seed=args.seed,
        use_cpu=args.device == "cpu",
        disable_tqdm=False,
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_final_artifacts(model, tokenizer, output_dir, *, merge_adapters, save_adapters_final):
    output_dir = Path(output_dir)
    adapters_dir = None
    if save_adapters_final or not merge_adapters:
        adapters_dir = output_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapters_dir)
        tokenizer.save_pretrained(adapters_dir)

    merged_dir = None
    if merge_adapters:
        if next(model.parameters()).device.type != "cpu":
            model = model.to("cpu")
        merged_model = model.merge_and_unload()
        merged_model.config.use_cache = True
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
    return adapters_dir, merged_dir


def main():
    args = parse_args()
    args.device = validate_device(args.device)
    configure_reproducibility(args.seed, deterministic=False)

    if args.pin_memory is None:
        args.pin_memory = args.device == "cuda"
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    wandb_mode = choose_wandb_mode(args.wandb_mode)
    os.environ["WANDB_MODE"] = wandb_mode
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    model, tokenizer, target_modules = load_model_and_tokenizer(args)
    train_dataset, eval_dataset = build_datasets(args, tokenizer)
    report_to = [] if wandb_mode == "disabled" else ["wandb"]
    trainer = Trainer(
        model=model,
        args=training_arguments(args, report_to=report_to),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "device": args.device,
                "dtype": args.dtype,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
                "train_skipped": train_dataset.skipped,
                "eval_skipped": eval_dataset.skipped if eval_dataset is not None else 0,
                "lora_target_modules": target_modules,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            indent=2,
        )
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    train_metrics = dict(train_result.metrics)
    if "train_runtime" in train_metrics and train_metrics["train_runtime"] > 0:
        train_metrics["train_tokens_per_second_estimate"] = (
            args.batch_size * args.grad_accumulation * args.max_length / train_metrics["train_runtime"]
        )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        if isinstance(eval_loss, (float, int)) and math.isfinite(float(eval_loss)):
            eval_metrics["eval_perplexity"] = math.exp(float(eval_loss))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    run_summary = {
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "output_dir": str(args.output_dir),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "device": args.device,
        "dtype": args.dtype,
        "wandb_mode": wandb_mode,
        "lora_target_modules": target_modules,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "train_skipped": train_dataset.skipped,
        "eval_skipped": eval_dataset.skipped if eval_dataset is not None else 0,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    adapters_dir = None
    merged_dir = None
    if args.save_final:
        adapters_dir, merged_dir = save_final_artifacts(
            trainer.model,
            tokenizer,
            args.output_dir,
            merge_adapters=args.merge_adapters,
            save_adapters_final=args.save_adapters_final,
        )
        run_summary["adapters_dir"] = str(adapters_dir) if adapters_dir is not None else None
        run_summary["merged_dir"] = str(merged_dir) if merged_dir is not None else None

    write_json(Path(args.output_dir) / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
