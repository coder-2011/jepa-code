from argparse import ArgumentParser
from pathlib import Path
import os
import random
import sys

import torch
import wandb
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.data import create_fineweb_dataloader
from text_jepa.models.layer_model import LayerModel
from text_jepa.tokenization import load_tokenizer_from_yaml
from text_jepa.train.step import train_step


def parse_args():
    parser = ArgumentParser()
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
    return parser.parse_args()


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_wandb_mode(requested_mode):
    if requested_mode is not None:
        return requested_mode
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def build_model(config, tokenizer, predictor_num_layers):
    model_config = config["model"]
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
        "max_length": config["tokenizer"]["max_length"],
        "hidden_dim": model_config["hidden_dim"],
        "num_heads": model_config["num_heads"],
        "num_layers": model_config["num_layers"],
        "ffn_dim": model_config["ffn_dim"],
        "dropout": model_config["dropout"],
        "ema_momentum": model_config["ema_momentum"],
        "dataset_size": dataset_size,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    config = load_config(args.config)
    tokenizer = load_tokenizer_from_yaml(args.config)
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
    dataset_size = len(dataloader.dataset)
    if dataset_size == 0:
        raise ValueError("No documents matched the current FineWeb dataloader filters")

    model = build_model(config, tokenizer, args.predictor_num_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=choose_wandb_mode(args.wandb_mode),
        config=build_run_config(args, config, dataset_size),
    ) as run:
        step = 0
        while step < args.steps:
            for batch in dataloader:
                batch = move_batch_to_device(batch, args.device)
                outputs = train_step(model, optimizer, batch)
                loss_value = outputs["loss"].item()

                step += 1
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

                if step >= args.steps:
                    break


if __name__ == "__main__":
    main()
