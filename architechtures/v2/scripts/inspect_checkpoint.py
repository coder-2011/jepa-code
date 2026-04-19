from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from data.dataset_helpers import build_eval_dataloader, dataset_dir_for_variant, list_split_shards
from intertwined_hjepa import IntertwinedConfig, IntertwinedHJEPA, warmup_weight


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ["Once upon a time", "The meaning of life is", "In a small town"]
EVAL_KEYS = ("loss", "loss_lm", "loss_jepa", "loss_sigreg")
LAYER_KEYS = (
    "loss_jepa",
    "loss_sigreg",
    "z_std_mean",
    "z_std_min",
    "z_std_max",
    "z_abs_max",
    "z_effective_rank",
    "delta_norm",
    "target_delta_norm",
    "delta_target_ratio",
)
RESIDUAL_KEYS = (
    "input_norm",
    "post_attn_norm",
    "output_norm",
    "attn_update_norm",
    "block_update_norm",
    "total_update_norm",
    "attn_update_ratio",
    "block_update_ratio",
    "total_update_ratio",
    "attn_block_cosine",
)


@dataclass(frozen=True)
class ParameterGolfAssets:
    dataset_root: Path
    tokenizer_path: Path
    vocab_size: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an Intertwined H-JEPA checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parameter-golf-root", type=Path, required=True)
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to sample from; repeatable")
    args = parser.parse_args(argv)
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.eval_batches > 0, "eval_batches must be positive"
    assert args.max_new_tokens > 0, "max_new_tokens must be positive"
    assert args.temperature > 0, "temperature must be positive"
    assert args.top_k >= 0, "top_k must be non-negative"
    return args


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_parameter_golf_assets(parameter_golf_root: Path, variant: str) -> ParameterGolfAssets:
    root = parameter_golf_root.expanduser().resolve()
    manifest_path = root / "data" / "manifest.json"
    assert manifest_path.is_file(), f"Missing parameter-golf manifest: {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = dataset_dir_for_variant(variant)
    dataset_by_name = {entry["name"]: entry for entry in manifest["datasets"]}
    tokenizer_by_name = {entry["name"]: entry for entry in manifest["tokenizers"]}
    dataset_entry = dataset_by_name[dataset_name]
    tokenizer_entry = tokenizer_by_name[dataset_entry["tokenizer_name"]]

    dataset_root = root / "data" / dataset_entry["path"]
    tokenizer_path = root / "data" / tokenizer_entry["model_path"]
    assert dataset_root.is_dir(), f"Missing dataset root: {dataset_root}"
    assert tokenizer_path.is_file(), f"Missing tokenizer model: {tokenizer_path}"
    return ParameterGolfAssets(dataset_root, tokenizer_path, int(dataset_entry["vocab_size"]))


def load_checkpoint(path: Path, device: torch.device) -> tuple[IntertwinedHJEPA, dict[str, Any], list[str]]:
    assert path.is_file(), f"Checkpoint not found: {path}"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    assert "config" in checkpoint and "model" in checkpoint, "Checkpoint must contain config and model"

    yaml_config = IntertwinedConfig.from_yaml(ROOT / "intertwined_hjepa.yaml")
    config_values = dict(checkpoint["config"])
    missing_fields = [name for name in yaml_config.__dataclass_fields__ if name not in config_values]
    for name in missing_fields:
        config_values[name] = getattr(yaml_config, name)
    config = IntertwinedConfig(**config_values)
    model = IntertwinedHJEPA(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, missing_fields


def move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, labels = batch
    return input_ids.to(device), labels.to(device)


def effective_rank(z: torch.Tensor) -> float:
    z = z.float().reshape(-1, z.shape[-1])
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.cpu())
    total = singular_values.sum().clamp_min(1e-12)
    probs = singular_values / total
    return float(torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum()))


def new_metric_totals(keys: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(keys, 0.0)


def average_metrics(metrics: dict[str, float], count: int) -> dict[str, float]:
    assert count > 0, "cannot average over zero items"
    return {key: value / count for key, value in metrics.items()}


def add_layer_metrics(
    layer: dict[str, float],
    outputs: dict[str, Any],
    index: int,
) -> None:
    z = outputs["z"][index].detach().float()
    delta = outputs["deltas"][index].detach().float()
    target_delta = outputs["targets"][index].detach().float() - z
    jepa_valid_mask = outputs["jepa_valid_mask"].to(torch.bool)
    z_flat = z.reshape(-1, z.shape[-1])
    z_std = z_flat.std(dim=0, unbiased=False)
    delta_norm = delta[jepa_valid_mask].norm(dim=-1).mean()
    target_delta_norm = target_delta[jepa_valid_mask].norm(dim=-1).mean()

    layer["loss_jepa"] += outputs["loss_jepa_layers"][index].item()
    layer["loss_sigreg"] += outputs["loss_sigreg_layers"][index].item()
    layer["z_std_mean"] += z_std.mean().item()
    layer["z_std_min"] += z_std.min().item()
    layer["z_std_max"] += z_std.max().item()
    layer["z_abs_max"] += z.abs().max().item()
    layer["z_effective_rank"] += effective_rank(z)
    layer["delta_norm"] += delta_norm.item()
    layer["target_delta_norm"] += target_delta_norm.item()
    layer["delta_target_ratio"] += (delta_norm / target_delta_norm.clamp_min(1e-12)).item()


def mean_token_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 3, "expected (B, L, D) tensor"
    return x.detach().float().norm(dim=-1).mean()


def mean_centered_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    assert left.shape == right.shape, "cosine inputs must have matching shape"
    left_flat = left.detach().float().reshape(-1, left.shape[-1])
    right_flat = right.detach().float().reshape(-1, right.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=-1, eps=1e-8)
    return float(cosine.mean())


def add_residual_metrics(layer: dict[str, float], outputs: dict[str, Any], index: int) -> None:
    state_in = outputs["states"][index].detach().float()
    post_attn = outputs["post_attn_states"][index].detach().float()
    state_out = outputs["states"][index + 1].detach().float()
    attn_update = post_attn - state_in
    block_update = state_out - post_attn
    total_update = state_out - state_in

    input_norm = mean_token_norm(state_in)
    post_attn_norm = mean_token_norm(post_attn)
    output_norm = mean_token_norm(state_out)
    attn_update_norm = mean_token_norm(attn_update)
    block_update_norm = mean_token_norm(block_update)
    total_update_norm = mean_token_norm(total_update)

    layer["input_norm"] += input_norm.item()
    layer["post_attn_norm"] += post_attn_norm.item()
    layer["output_norm"] += output_norm.item()
    layer["attn_update_norm"] += attn_update_norm.item()
    layer["block_update_norm"] += block_update_norm.item()
    layer["total_update_norm"] += total_update_norm.item()
    layer["attn_update_ratio"] += (attn_update_norm / input_norm.clamp_min(1e-12)).item()
    layer["block_update_ratio"] += (block_update_norm / post_attn_norm.clamp_min(1e-12)).item()
    layer["total_update_ratio"] += (total_update_norm / input_norm.clamp_min(1e-12)).item()
    layer["attn_block_cosine"] += mean_centered_cosine(attn_update, block_update)


def inspect_eval(
    model: IntertwinedHJEPA,
    loader,
    *,
    device: torch.device,
    step: int,
) -> tuple[dict[str, float], list[dict[str, float]], list[str]]:
    config = model.config
    totals = new_metric_totals(EVAL_KEYS)
    layer_totals = [new_metric_totals(LAYER_KEYS) for _ in range(config.depth - 1)]
    residual_totals = [new_metric_totals(RESIDUAL_KEYS) for _ in range(config.depth)]
    warnings: list[str] = []
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels = move_batch(batch, device)
            assert int(input_ids.max()) < config.vocab_size, "input_ids contain token ids >= config.vocab_size"
            assert int(labels.max()) < config.vocab_size, "labels contain token ids >= config.vocab_size"

            outputs = model(input_ids=input_ids, labels=labels, step=step)
            assert torch.isfinite(outputs["loss"]), "non-finite total loss"

            for key in EVAL_KEYS:
                source_key = "loss_main" if key == "loss_lm" else key
                totals[key] += outputs[source_key].item()

            for index in range(config.depth - 1):
                add_layer_metrics(layer_totals[index], outputs, index)
            for index in range(config.depth):
                add_residual_metrics(residual_totals[index], outputs, index)

            num_batches += 1

    totals = average_metrics(totals, num_batches)
    layer_totals = [average_metrics(layer, num_batches) for layer in layer_totals]
    residual_totals = [average_metrics(layer, num_batches) for layer in residual_totals]

    lambda_eff = warmup_weight(config.lambda_jepa, step, config.jepa_warmup_steps)
    beta_eff = warmup_weight(config.beta_sigreg, step, config.sigreg_warmup_steps)
    totals["lambda_eff"] = lambda_eff
    totals["beta_eff"] = beta_eff
    totals["jepa_contribution"] = lambda_eff * totals["loss_jepa"]
    totals["sigreg_contribution"] = beta_eff * totals["loss_sigreg"]

    for index, layer in enumerate(layer_totals):
        if layer["z_std_min"] < 1e-4:
            warnings.append(f"layer {index} z_std_min is very small: {layer['z_std_min']:.3e}")
        if layer["delta_target_ratio"] < 0.01 or layer["delta_target_ratio"] > 100:
            warnings.append(
                f"layer {index} delta/target ratio looks off: {layer['delta_target_ratio']:.4f}"
            )
    for index, layer in enumerate(residual_totals):
        block_name = "final" if index == config.depth - 1 else f"jepa {index}"
        if layer["attn_update_ratio"] < 1e-4:
            warnings.append(f"{block_name} attention update is very small: {layer['attn_update_ratio']:.3e}")
        if layer["block_update_ratio"] < 1e-4:
            warnings.append(f"{block_name} block update is very small: {layer['block_update_ratio']:.3e}")

    return totals, layer_totals, residual_totals, warnings


def sample_next_id(logits: torch.Tensor, *, temperature: float, top_k: int) -> int:
    assert logits.ndim == 1, "next-token logits must be one-dimensional"
    assert torch.isfinite(logits).all(), "next-token logits contain non-finite values"
    logits = logits.float() / temperature
    if top_k <= 0:
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()

    values, indices = torch.topk(logits, min(top_k, logits.numel()))
    selected = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1)
    return int(indices[selected].item())


def token_repetition_stats(token_ids: list[int]) -> dict[str, float | int]:
    assert token_ids, "token_ids must not be empty"

    adjacent_repeats = sum(left == right for left, right in zip(token_ids, token_ids[1:]))
    run_length = 1
    max_run = 1
    for left, right in zip(token_ids, token_ids[1:]):
        if left == right:
            run_length += 1
            max_run = max(max_run, run_length)
        else:
            run_length = 1

    return {
        "unique_new_tokens": len(set(token_ids)),
        "repeated_fraction": adjacent_repeats / max(1, len(token_ids) - 1),
        "max_run": max_run,
    }


@torch.no_grad()
def generate(
    model: IntertwinedHJEPA,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, out_type=int)
    assert ids, f"Prompt encoded to no tokens: {prompt!r}"
    assert max(ids) < model.config.vocab_size, "prompt contains token ids >= config.vocab_size"

    generated = list(ids)
    for _ in range(max_new_tokens):
        context = generated[-model.config.max_length :]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids=x, compute_aux_losses=False)["logits"][0, -1]
        generated.append(sample_next_id(logits, temperature=temperature, top_k=top_k))

    new_ids = generated[len(ids) :]
    return {
        "prompt": prompt,
        "prompt_ids": ids,
        "new_ids": new_ids,
        "decoded": tokenizer.decode(generated),
        **token_repetition_stats(new_ids),
    }


def generation_rows(item: dict[str, Any]) -> list[str]:
    return [
        f"prompt: {item['prompt']!r}",
        f"prompt_ids: {item['prompt_ids']}",
        f"new_ids: {item['new_ids']}",
        f"decoded: {item['decoded']!r}",
        (
            f"unique_new_tokens={item['unique_new_tokens']} "
            f"repeated_fraction={item['repeated_fraction']:.3f} "
            f"max_run={item['max_run']}"
        ),
    ]


def print_section(title: str, rows: list[str]) -> None:
    print(title)
    for row in rows:
        print(f"  {row}")
    print()


def print_report(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config: IntertwinedConfig,
    dataset_root: Path,
    tokenizer_path: Path,
    eval_metrics: dict[str, float],
    layer_metrics: list[dict[str, float]],
    residual_metrics: list[dict[str, float]],
    generations: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    print_section(
        "Checkpoint",
        [
            f"path: {checkpoint_path}",
            f"step: {checkpoint.get('step')}",
            f"tokens_processed: {checkpoint.get('tokens_processed')}",
            f"dataset_root: {dataset_root}",
            f"tokenizer: {tokenizer_path}",
            f"vocab_size: {config.vocab_size}",
            f"max_length: {config.max_length}",
        ],
    )

    eval_keys = EVAL_KEYS + ("lambda_eff", "beta_eff", "jepa_contribution", "sigreg_contribution")
    print_section("Eval", [f"{key}: {eval_metrics[key]:.6f}" for key in eval_keys])

    print_section(
        "Layer Diagnostics",
        [
            (
                f"layer {index}: "
                f"jepa={layer['loss_jepa']:.6f} "
                f"sigreg={layer['loss_sigreg']:.6f} "
                f"z_std_mean={layer['z_std_mean']:.6f} "
                f"z_std_min={layer['z_std_min']:.6f} "
                f"eff_rank={layer['z_effective_rank']:.2f} "
                f"delta_norm={layer['delta_norm']:.6f} "
                f"target_delta_norm={layer['target_delta_norm']:.6f} "
                f"delta/target={layer['delta_target_ratio']:.6f}"
            )
            for index, layer in enumerate(layer_metrics)
        ],
    )

    print_section(
        "Residual Stream",
        [
            (
                f"layer {index if index < len(residual_metrics) - 1 else 'final'}: "
                f"in={layer['input_norm']:.6f} "
                f"post_attn={layer['post_attn_norm']:.6f} "
                f"out={layer['output_norm']:.6f} "
                f"attn_update={layer['attn_update_norm']:.6f} "
                f"block_update={layer['block_update_norm']:.6f} "
                f"total_update={layer['total_update_norm']:.6f} "
                f"attn/input={layer['attn_update_ratio']:.6f} "
                f"block/post={layer['block_update_ratio']:.6f} "
                f"total/input={layer['total_update_ratio']:.6f} "
                f"cos(attn,block)={layer['attn_block_cosine']:.6f}"
            )
            for index, layer in enumerate(residual_metrics)
        ],
    )

    print("Generation")
    for item in generations:
        for row in generation_rows(item):
            print(f"  {row}")
        print()

    print("Warnings")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  none")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or default_device())
    model, checkpoint, missing_config_fields = load_checkpoint(args.checkpoint, device)
    config = model.config
    seq_len = config.max_length if args.seq_len is None else args.seq_len

    assets = resolve_parameter_golf_assets(
        args.parameter_golf_root,
        args.variant,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(assets.tokenizer_path))
    tokenizer_vocab_size = tokenizer.get_piece_size()
    assert config.vocab_size == assets.vocab_size == tokenizer_vocab_size, (
        f"vocab mismatch: config={config.vocab_size}, "
        f"dataset={assets.vocab_size}, tokenizer={tokenizer_vocab_size}"
    )

    eval_loader = build_eval_dataloader(
        list_split_shards(assets.dataset_root, "val"),
        batch_size=args.batch_size,
        seq_len=seq_len,
        max_batches=args.eval_batches,
        pin_memory=device.type == "cuda",
    )
    step = int(checkpoint.get("step", 0))
    eval_metrics, layer_metrics, residual_metrics, warnings = inspect_eval(model, eval_loader, device=device, step=step)
    if missing_config_fields:
        warnings.append(
            "checkpoint config was missing fields filled from current YAML: "
            + ", ".join(missing_config_fields)
        )

    prompts = args.prompt or DEFAULT_PROMPTS
    generations = [
        generate(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        for prompt in prompts
    ]
    for item in generations:
        if item["repeated_fraction"] > 0.5:
            warnings.append(f"high repetition for prompt {item['prompt']!r}")
        if item["max_run"] >= 8:
            warnings.append(f"long token run for prompt {item['prompt']!r}: {item['max_run']}")

    print_report(
        args.checkpoint,
        checkpoint,
        config,
        assets.dataset_root,
        assets.tokenizer_path,
        eval_metrics,
        layer_metrics,
        residual_metrics,
        generations,
        warnings,
    )


if __name__ == "__main__":
    main()
