from argparse import ArgumentParser
import math
from pathlib import Path
import os
import sys

import torch
import wandb
from transformers import AutoModelForCausalLM
from wandb import AlertLevel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.env import load_local_env
from text_jepa.data import create_llm_jepa_dataloader
from text_jepa.models.llm_jepa import LLMJEPAModel
from text_jepa.tokenization import load_tokenizer_from_yaml, load_yaml_config
from text_jepa.train.llm_jepa_step import train_llm_jepa_step
from text_jepa.utils.repro import configure_reproducibility, resolve_deterministic, resolve_seed

load_local_env(ROOT)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "text-jepa-default.yaml"))
    parser.add_argument(
        "--train-file",
        default=str(ROOT / "llm-jepa" / "datasets" / "synth_train.jsonl"),
    )
    parser.add_argument(
        "--eval-file",
        default=str(ROOT / "llm-jepa" / "datasets" / "synth_test.jsonl"),
    )
    parser.add_argument("--model-name")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--predictors", type=int, default=1)
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--gamma-lm", type=float, default=1.0)
    parser.add_argument("--jepa-metric", default="cosine")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--wandb-project", default="llm-jepa")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "llm-jepa"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-from")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=2)
    return parser.parse_args()


def choose_wandb_mode(requested_mode):
    if requested_mode is not None:
        return requested_mode
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def move_batch_to_device(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def build_model(args, config, tokenizer):
    model_name = args.model_name or config["tokenizer"]["model_name"]
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    backbone.resize_token_embeddings(len(tokenizer))
    return LLMJEPAModel(
        backbone,
        lambda_jepa=args.lambda_jepa,
        gamma_lm=args.gamma_lm,
        jepa_metric=args.jepa_metric,
    )


def build_run_config(args, config, dataset_size, max_length):
    return {
        "config_path": args.config,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "model_name": args.model_name or config["tokenizer"]["model_name"],
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "predictors": args.predictors,
        "lambda_jepa": args.lambda_jepa,
        "gamma_lm": args.gamma_lm,
        "jepa_metric": args.jepa_metric,
        "checkpoint_dir": args.checkpoint_dir,
        "save_every": args.save_every,
        "resume_from": args.resume_from,
        "auto_resume": args.auto_resume,
        "val_every": args.val_every,
        "val_max_batches": args.val_max_batches,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": args.device,
        "max_length": max_length,
        "dataset_size": dataset_size,
    }


def checkpoint_state(step, model, optimizer, run_config):
    return {
        "step": step,
        "model": model.state_dict(),
        "config": run_config,
    }


def save_checkpoint(checkpoint_dir, step, state):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_path = checkpoint_dir / f"step-{step:06d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    tmp_path = step_path.with_suffix(step_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, step_path)
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    os.link(step_path, latest_path)
    return step_path


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def resolve_resume_path(resume_from, checkpoint_dir, auto_resume):
    if resume_from is not None:
        return Path(resume_from)
    if not auto_resume:
        return None
    latest_path = Path(checkpoint_dir) / "latest.pt"
    if latest_path.exists():
        return latest_path
    return None


def evaluate(model, dataloader, device, max_batches):
    model.eval()
    totals = {"loss": 0.0, "lm_loss": 0.0, "jepa_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            totals["loss"] += outputs["loss"].item()
            totals["lm_loss"] += outputs["lm_loss"].item()
            totals["jepa_loss"] += outputs["jepa_loss"].item()
            count += 1
    model.train()
    if count == 0:
        raise ValueError("evaluation dataloader produced no batches")
    return {name: value / count for name, value in totals.items()}


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    args.seed = resolve_seed(config, args.seed)
    args.deterministic = resolve_deterministic(config, args.deterministic)
    configure_reproducibility(args.seed, deterministic=args.deterministic)
    tokenizer = load_tokenizer_from_yaml(args.config)
    max_length = args.max_length or config["tokenizer"]["max_length"]

    train_dataloader = create_llm_jepa_dataloader(
        jsonl_path=args.train_file,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=args.batch_size,
        predictors=args.predictors,
        shuffle=True,
        seed=args.seed,
        max_docs=args.max_docs,
    )
    eval_dataloader = None
    if args.eval_file and args.val_every > 0:
        eval_dataloader = create_llm_jepa_dataloader(
            jsonl_path=args.eval_file,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
            predictors=args.predictors,
            shuffle=False,
            seed=args.seed + 10_000,
            max_docs=args.max_docs,
        )

    dataset_size = len(train_dataloader.dataset)
    if dataset_size == 0:
        raise ValueError("No LLM-JEPA training examples were loaded")

    model = build_model(args, config, tokenizer).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    wandb_mode = choose_wandb_mode(args.wandb_mode)
    run_config = build_run_config(args, config, dataset_size, max_length)
    start_step = 0
    resume_path = resolve_resume_path(args.resume_from, args.checkpoint_dir, args.auto_resume)
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, model, optimizer, args.device)
        message = "auto-resumed" if args.resume_from is None else "resumed"
        print(f"{message} from {resume_path} at step={start_step}")
    else:
        print("starting fresh")

    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=wandb_mode,
        config=run_config,
    ) as run:
        step = start_step
        try:
            while step < args.steps:
                for batch in train_dataloader:
                    batch = move_batch_to_device(batch, args.device)
                    outputs = train_llm_jepa_step(model, optimizer, batch)
                    loss_value = outputs["loss"].item()
                    if not math.isfinite(loss_value):
                        if wandb_mode not in {"offline", "disabled"}:
                            run.alert(
                                title="LLM-JEPA training loss is non-finite",
                                text=f"step={step + 1} loss={loss_value}",
                                level=AlertLevel.ERROR,
                                wait_duration=300,
                            )
                        raise ValueError(f"loss became non-finite at step {step + 1}: {loss_value}")

                    step += 1
                    run.log(
                        {
                            "train/loss": loss_value,
                            "train/lm_loss": outputs["lm_loss"].item(),
                            "train/jepa_loss": outputs["jepa_loss"].item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step": step,
                            "train/batch_size": batch["input_ids"].shape[0],
                        },
                        step=step,
                    )
                    print(
                        "step="
                        f"{step} loss={loss_value:.6f} "
                        f"lm_loss={outputs['lm_loss'].item():.6f} "
                        f"jepa_loss={outputs['jepa_loss'].item():.6f}"
                    )

                    if args.save_every > 0 and step % args.save_every == 0:
                        save_checkpoint(
                            args.checkpoint_dir,
                            step,
                            checkpoint_state(step, model, optimizer, run_config),
                        )
                    if eval_dataloader is not None and step % args.val_every == 0:
                        metrics = evaluate(model, eval_dataloader, args.device, args.val_max_batches)
                        run.log(
                            {
                                "val/loss": metrics["loss"],
                                "val/lm_loss": metrics["lm_loss"],
                                "val/jepa_loss": metrics["jepa_loss"],
                            },
                            step=step,
                        )
                        print(
                            "step="
                            f"{step} val_loss={metrics['loss']:.6f} "
                            f"val_lm_loss={metrics['lm_loss']:.6f} "
                            f"val_jepa_loss={metrics['jepa_loss']:.6f}"
                        )

                    if step >= args.steps:
                        break
            if step > 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    step,
                    checkpoint_state(step, model, optimizer, run_config),
                )
        except Exception as error:
            if wandb_mode not in {"offline", "disabled"}:
                run.alert(
                    title="LLM-JEPA training run crashed",
                    text=str(error),
                    level=AlertLevel.ERROR,
                    wait_duration=300,
                )
            raise


if __name__ == "__main__":
    main()
