from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_jepa.benchmarking import (  # noqa: E402
    append_result,
    benchmark_messages,
    dataset_task_name,
    gold_text,
    load_existing,
    load_rows,
    prompt_text,
    score_prediction,
)
from text_jepa.data import ensure_predictor_tokens  # noqa: E402
from text_jepa.env import load_local_env  # noqa: E402
from text_jepa.models.llm_jepa import LLMJEPAModel  # noqa: E402
from text_jepa.tokenization import load_tokenizer_from_yaml, load_yaml_config  # noqa: E402


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CONFIG = ROOT / "text-jepa-default.yaml"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "llm-jepa" / "latest.pt"
DEFAULT_DATASET = ROOT / "llm-jepa" / "datasets" / "synth_test.jsonl"
DEFAULT_OUTPUT = ROOT / "tmp" / "local_benchmark_results.jsonl"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4"
SYNTH_JUDGE_PROMPT = (
    "You are grading a regex answer. Decide whether the candidate regex is functionally correct for "
    "the natural-language request. Ignore formatting differences and minor syntax variation if the "
    "candidate expresses the same condition. Also score how close the candidate is overall on a 0-100 scale, "
    "where 100 means functionally equivalent to the intended regex and 0 means unrelated or unusable. "
    "Return JSON with keys correct (boolean), closeness (integer 0-100), and reason (string)."
)


class RateLimitExceeded(RuntimeError):
    pass


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def request_openrouter(
    session: requests.Session,
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    response_format: dict | None = None,
    disable_reasoning: bool = True,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if disable_reasoning:
        payload["reasoning"] = {"effort": "none", "exclude": True}
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.jepa-code",
        "X-OpenRouter-Title": "jepa-code local benchmark",
    }

    for attempt in range(retries):
        response = session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 429:
            if attempt + 1 >= retries:
                raise RateLimitExceeded(response.text)
            retry_after = response.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else retry_sleep * (attempt + 1)
            time.sleep(sleep_seconds)
            continue
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    raise RateLimitExceeded("OpenRouter rate limit retries exhausted")


def judge_synth_prediction(
    session: requests.Session,
    *,
    api_key: str,
    judge_model: str,
    row: dict,
    prediction: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> tuple[bool | None, int | None, str | None]:
    raw = request_openrouter(
        session,
        api_key=api_key,
        model=judge_model,
        messages=[
            {"role": "system", "content": SYNTH_JUDGE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": row["messages"][1]["content"],
                        "gold_regex": row["messages"][2]["content"],
                        "candidate_regex": prediction,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        max_tokens=128,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
        response_format={"type": "json_object"},
        disable_reasoning=False,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, None, f"judge_invalid_json: {raw}"
    correct = parsed.get("correct")
    closeness = parsed.get("closeness")
    reason = parsed.get("reason")
    normalized_closeness = None
    if isinstance(closeness, (int, float)):
        normalized_closeness = max(0, min(100, int(round(closeness))))
    return (
        bool(correct) if isinstance(correct, bool) else None,
        normalized_closeness,
        reason if isinstance(reason, str) else None,
    )


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required in the environment or .env.local")
    return api_key


def build_tokenizer(config_path: Path, predictors: int):
    config = load_yaml_config(config_path)
    tokenizer = load_tokenizer_from_yaml(str(config_path))
    ensure_predictor_tokens(tokenizer, predictors)
    return tokenizer, config


def load_local_model(args: argparse.Namespace):
    checkpoint = None
    checkpoint_config = {}
    config_path = Path(args.config)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config") or {}
        config_path = Path(checkpoint_config.get("config_path") or args.config)
        trained_model_name = checkpoint_config.get("model_name")
        if trained_model_name and trained_model_name != args.base_model:
            raise ValueError(
                f"checkpoint {checkpoint_path} was trained on {trained_model_name}, "
                f"not {args.base_model}; point --checkpoint at the real fine-tuned {args.base_model} run"
            )

    predictors = int(checkpoint_config.get("predictors", args.predictors))
    tokenizer, _ = build_tokenizer(config_path, predictors)
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    backbone.resize_token_embeddings(len(tokenizer))

    if checkpoint is None:
        backbone.to(args.device)
        backbone.eval()
        return backbone, tokenizer, False

    model = LLMJEPAModel(backbone)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()
    return model, tokenizer, True


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def clean_generation_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def maybe_disable_qwen_thinking(messages: list[dict[str, str]], base_model: str) -> list[dict[str, str]]:
    if "Qwen/Qwen3" not in base_model:
        return messages
    adjusted = [dict(message) for message in messages]
    directive = "/no_think"
    if adjusted and adjusted[0]["role"] == "system":
        if directive not in adjusted[0]["content"]:
            adjusted[0]["content"] = directive + "\n" + adjusted[0]["content"]
        return adjusted
    return [{"role": "system", "content": directive}] + adjusted


def select_hellaswag_choice(
    model,
    *,
    is_llm_jepa_checkpoint: bool,
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    device: str,
) -> str:
    inputs = tokenizer(
        render_prompt(tokenizer, messages),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    scoring_model = model.backbone if is_llm_jepa_checkpoint else model
    with torch.no_grad():
        outputs = scoring_model(**inputs)
    logits = outputs.logits[0, -1]

    scores = []
    for choice in ("A", "B", "C", "D"):
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"HellaSwag choice {choice!r} is not a single token for this tokenizer")
        scores.append((choice, logits[token_ids[0]].item()))
    return max(scores, key=lambda item: item[1])[0]


def local_generate(
    model,
    *,
    is_llm_jepa_checkpoint: bool,
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    max_new_tokens: int,
    device: str,
) -> str:
    inputs = tokenizer(
        render_prompt(tokenizer, messages),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    generation_model = model.backbone if is_llm_jepa_checkpoint else model
    with torch.no_grad():
        outputs = generation_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return clean_generation_text(tokenizer.decode(generated, skip_special_tokens=True))


def benchmark(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output_path.exists():
        output_path.unlink()

    rows = load_rows(dataset_path, args.max_examples)
    existing = {} if args.force else load_existing(output_path)
    model, tokenizer, is_llm_jepa_checkpoint = load_local_model(args)
    task_name = dataset_task_name(dataset_path.name)
    api_key = require_openrouter_api_key() if task_name == "synth" else None
    session = requests.Session()

    completed = 0
    correct = 0
    exact_correct = 0
    relaxed_correct = 0
    relaxed_scored = 0
    closeness_total = 0
    closeness_scored = 0
    failed = 0
    samples: list[dict] = []
    stopped_reason = None

    with output_path.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            if index in existing:
                result = existing[index]
            else:
                try:
                    prompt_messages = benchmark_messages(
                        row["messages"],
                        dataset_name=dataset_path.name,
                        strict_answer_only=not args.no_strict_answer_only,
                    )
                    prompt_messages = maybe_disable_qwen_thinking(prompt_messages, args.base_model)
                    if task_name == "hellaswag":
                        prediction = select_hellaswag_choice(
                            model,
                            tokenizer=tokenizer,
                            messages=prompt_messages,
                            is_llm_jepa_checkpoint=is_llm_jepa_checkpoint,
                            max_length=args.max_length,
                            device=args.device,
                        )
                    else:
                        prediction = local_generate(
                            model,
                            tokenizer=tokenizer,
                            messages=prompt_messages,
                            is_llm_jepa_checkpoint=is_llm_jepa_checkpoint,
                            max_length=args.max_length,
                            max_new_tokens=args.max_new_tokens,
                            device=args.device,
                        )
                    result = {
                        "index": index,
                        "prompt": prompt_text(row),
                        "gold": gold_text(row),
                        "prediction": prediction,
                        "match_exact": score_prediction(
                            prediction,
                            row,
                            dataset_path.name,
                            spider_path=args.spider_path,
                            startswith=args.startswith,
                        ),
                        "match": None,
                        "match_relaxed": None,
                        "closeness": None,
                        "judge_reason": None,
                        "error": None,
                    }
                    if task_name == "synth":
                        judged, closeness, reason = judge_synth_prediction(
                            session,
                            api_key=api_key,
                            judge_model=args.judge_model,
                            row=row,
                            prediction=prediction,
                            timeout=args.timeout,
                            retries=args.retries,
                            retry_sleep=args.retry_sleep,
                        )
                        result["match_relaxed"] = judged
                        result["closeness"] = closeness
                        result["judge_reason"] = reason
                        result["match"] = judged
                    else:
                        result["match"] = result["match_exact"]
                except RateLimitExceeded as error:
                    stopped_reason = f"rate_limited: {error}"
                    break
                except Exception as error:  # noqa: BLE001
                    result = {
                        "index": index,
                        "prompt": prompt_text(row),
                        "gold": gold_text(row),
                        "prediction": "",
                        "match_exact": False,
                        "match": False,
                        "match_relaxed": None,
                        "closeness": None,
                        "judge_reason": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                append_result(sink, result)

            completed += 1
            correct += int(bool(result.get("match")))
            exact_correct += int(bool(result.get("match_exact")))
            if result.get("match_relaxed") is not None:
                relaxed_scored += 1
                relaxed_correct += int(bool(result.get("match_relaxed")))
            if result.get("closeness") is not None:
                closeness_scored += 1
                closeness_total += int(result["closeness"])
            if result.get("error"):
                failed += 1
            if len(samples) < 5:
                samples.append(result)

    return {
        "dataset": str(dataset_path),
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "judge_model": args.judge_model,
        "count": completed,
        "correct": correct,
        "accuracy": (correct / completed) if completed else 0.0,
        "correct_exact": exact_correct,
        "accuracy_exact": (exact_correct / completed) if completed else 0.0,
        "correct_relaxed": relaxed_correct if relaxed_scored else None,
        "accuracy_relaxed": (relaxed_correct / relaxed_scored) if relaxed_scored else None,
        "relaxed_scored": relaxed_scored,
        "avg_closeness": (closeness_total / closeness_scored) if closeness_scored else None,
        "closeness_scored": closeness_scored,
        "failed": failed,
        "stopped_reason": stopped_reason,
        "output_file": str(output_path),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a local LLM-JEPA checkpoint on local datasets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--predictors", type=int, default=1)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-strict-answer-only", action="store_true")
    parser.add_argument("--spider-path", default=str(ROOT / "llm-jepa" / "spider_data" / "database"))
    parser.add_argument("--startswith", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_local_env(ROOT)
    print(json.dumps(benchmark(parse_args()), indent=2))


if __name__ == "__main__":
    main()
