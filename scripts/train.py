from argparse import ArgumentParser
import math
from pathlib import Path
import os
import random
import sys

import torch
import wandb
import yaml
from wandb import AlertLevel


ROOT = Path(__file__).resolve().parents[1]
# Keep script execution independent from editable installs while the package is still in active development.
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.data import create_fineweb_dataloader
from text_jepa.models.layer_model import LayerModel
from text_jepa.tokenization import load_tokenizer_from_yaml
from text_jepa.train.step import train_step


def parse_args():
    parser = ArgumentParser()
    # Defaults are intentionally small so the script can double as a local smoke-training entrypoint.
    parser.add_argument("--config", default=str(ROOT / "text-jepa-default.yaml"))
    parser.add_argument("--data-path", default=str(ROOT / "tmp" / "fineweb-sample.jsonl"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-token-count", type=int, default=256)
    parser.add_argument("--min-language-score", type=float, default=0.98)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--predictor-num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="layer-jepa")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "layer"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-from")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=2)
    return parser.parse_args()


def load_config(config_path):
    # The script needs the full nested config because it builds both tokenizer and model objects.
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_wandb_mode(requested_mode):
    if requested_mode is not None:
        return requested_mode
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    # Fall back to offline mode when no API key is present so local runs do not fail on logging setup.
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seed all visible CUDA devices for reproducible local experiments.
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    # The batch contract is tensor-only today, so a simple dict comprehension is enough.
    return {name: tensor.to(device) for name, tensor in batch.items()}


def build_model(config, tokenizer, predictor_num_layers):
    model_config = config["model"]
    # Reuse the encoder config for both towers and override only the predictor depth at the CLI.
    return LayerModel(
        vocab_size=len(tokenizer),
        max_length=config["tokenizer"]["max_length"],
        hidden_dim=model_config["hidden_dim"],
        encoder_num_layers=model_config["num_layers"],
        encoder_num_heads=model_config["num_heads"],
        encoder_ffn_dim=model_config["ffn_dim"],
        predictor_num_layers=predictor_num_layers,
        predictor_num_heads=model_config["num_heads"],
        predictor_ffn_dim=model_config["ffn_dim"],
        dropout=model_config["dropout"],
        ema_momentum=model_config["ema_momentum"],
    )


def build_run_config(args, config, dataset_size):
    model_config = config["model"]
    # Mirror the most important runtime knobs into W&B so checkpoints and metrics are interpretable later.
    return {
        "config_path": args.config,
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "min_token_count": args.min_token_count,
        "min_language_score": args.min_language_score,
        "max_docs": args.max_docs,
        "predictor_num_layers": args.predictor_num_layers,
        "seed": args.seed,
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "save_every": args.save_every,
        "resume_from": args.resume_from,
        "auto_resume": args.auto_resume,
        "val_every": args.val_every,
        "val_max_batches": args.val_max_batches,
        "max_length": config["tokenizer"]["max_length"],
        "hidden_dim": model_config["hidden_dim"],
        "num_heads": model_config["num_heads"],
        "num_layers": model_config["num_layers"],
        "ffn_dim": model_config["ffn_dim"],
        "dropout": model_config["dropout"],
        "ema_momentum": model_config["ema_momentum"],
        "dataset_size": dataset_size,
    }


def checkpoint_state(step, model, optimizer, run_config):
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": run_config,
    }


def save_checkpoint(checkpoint_dir, step, state):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_path = checkpoint_dir / f"step-{step:06d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    for path in (step_path, latest_path):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    return step_path


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
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
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            losses.append(model(**batch)["loss"].item())
    model.train()
    if not losses:
        raise ValueError("validation dataloader produced no batches")
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    seed_everything(args.seed)

    config = load_config(args.config)
    tokenizer = load_tokenizer_from_yaml(args.config)
    # Dataset filtering happens inside create_fineweb_dataloader so the training loop sees only ready batches.
    dataloader = create_fineweb_dataloader(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        config_path=args.config,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        min_token_count=args.min_token_count,
        min_language_score=args.min_language_score,
        max_docs=args.max_docs,
    )
    val_dataloader = None
    if args.val_every > 0:
        val_dataloader = create_fineweb_dataloader(
            jsonl_path=args.data_path,
            tokenizer=tokenizer,
            config_path=args.config,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 10_000,
            min_token_count=args.min_token_count,
            min_language_score=args.min_language_score,
            max_docs=args.max_docs,
        )
    dataset_size = len(dataloader.dataset)
    if dataset_size == 0:
        raise ValueError("No documents matched the current FineWeb dataloader filters")

    model = build_model(config, tokenizer, args.predictor_num_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    wandb_mode = choose_wandb_mode(args.wandb_mode)
    run_config = build_run_config(args, config, dataset_size)
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
                for batch in dataloader:
                    # Device transfer happens batch-by-batch so CPU dataloading stays simple.
                    batch = move_batch_to_device(batch, args.device)
                    outputs = train_step(model, optimizer, batch)
                    loss_value = outputs["loss"].item()
                    if not math.isfinite(loss_value):
                        if wandb_mode not in {"offline", "disabled"}:
                            run.alert(
                                title="Layer training loss is non-finite",
                                text=f"step={step + 1} loss={loss_value}",
                                level=AlertLevel.ERROR,
                                wait_duration=300,
                            )
                        raise ValueError(f"loss became non-finite at step {step + 1}: {loss_value}")

                    step += 1
                    # Log both optimization state and an explicit step count for easy charting.
                    run.log(
                        {
                            "train/loss": loss_value,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step": step,
                            "train/batch_size": batch["input_ids_full"].shape[0],
                        },
                        step=step,
                    )
                    print(f"step={step} loss={loss_value:.6f}")
                    if args.save_every > 0 and step % args.save_every == 0:
                        save_checkpoint(
                            args.checkpoint_dir,
                            step,
                            checkpoint_state(step, model, optimizer, run_config),
                        )
                    if val_dataloader is not None and step % args.val_every == 0:
                        val_loss = evaluate(model, val_dataloader, args.device, args.val_max_batches)
                        run.log({"val/loss": val_loss}, step=step)
                        print(f"step={step} val_loss={val_loss:.6f}")

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
                    title="Layer training run crashed",
                    text=str(error),
                    level=AlertLevel.ERROR,
                    wait_duration=300,
                )
            raise


if __name__ == "__main__":
    main()
