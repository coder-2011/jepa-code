from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-jepa"))

from evaluate import eval as official_eval  # noqa: E402


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"
DEFAULT_DATASET = ROOT / "llm-jepa" / "datasets" / "synth_test.jsonl"
DEFAULT_OUTPUT = ROOT / "tmp" / "openrouter_benchmark_results.jsonl"
ANSWER_ONLY = (
    "Return only the final answer. Do not explain. Do not use markdown. "
    "Do not include code fences. Do not include any extra text."
)
SYNTH_STYLE = (
    "Use the dataset's regex notation. Use ~ for negation and & for conjunction when needed. "
    "Do not add anchors like ^ or $. Return only the regex expression."
)


class RateLimitExceeded(RuntimeError):
    pass


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require_api_key() -> str:
    load_env(ROOT / ".env.local")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required in the environment or .env.local")
    return api_key


def load_rows(dataset_path: Path, max_examples: int | None) -> list[dict]:
    rows = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples is not None and index >= max_examples:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    return rows


def load_existing(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}
    existing = {}
    with output_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in row:
                existing[int(row["index"])] = row
    return existing


def benchmark_messages(
    messages: list[dict[str, str]],
    *,
    dataset_name: str,
    strict_answer_only: bool,
) -> list[dict[str, str]]:
    prompt_messages = [dict(message) for message in messages[:-1]]
    instructions = []
    if not strict_answer_only:
        return prompt_messages
    instructions.append(ANSWER_ONLY)
    if dataset_name.startswith("synth"):
        instructions.append(SYNTH_STYLE)
    if prompt_messages and prompt_messages[0]["role"] == "system":
        prompt_messages[0]["content"] = prompt_messages[0]["content"].rstrip() + "\n\n" + "\n".join(instructions)
    else:
        prompt_messages.insert(0, {"role": "system", "content": "\n".join(instructions)})
    return prompt_messages


def request_prediction(
    session: requests.Session,
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": {
            "effort": "none",
            "exclude": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.jepa-code",
        "X-OpenRouter-Title": "jepa-code benchmark",
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


def evaluate_prediction(prediction: str, row: dict, dataset_name: str) -> bool:
    return bool(
        official_eval(
            prediction,
            row["messages"],
            dataset_name,
            spider_path="",
            startswith=False,
            debug=0,
        )
    )


def append_result(handle, result: dict) -> None:
    handle.write(json.dumps(result) + "\n")
    handle.flush()


def benchmark(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output_path.exists():
        output_path.unlink()

    rows = load_rows(dataset_path, args.max_examples)
    existing = {} if args.force else load_existing(output_path)
    api_key = require_api_key()
    session = requests.Session()

    completed = 0
    correct = 0
    failed = 0
    samples: list[dict] = []
    stopped_reason = None

    with output_path.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            if index in existing:
                result = existing[index]
            else:
                try:
                    prediction = request_prediction(
                        session,
                        api_key=api_key,
                        model=args.model,
                        messages=benchmark_messages(
                            row["messages"],
                            dataset_name=dataset_path.name,
                            strict_answer_only=not args.no_strict_answer_only,
                        ),
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                        retry_sleep=args.retry_sleep,
                    )
                    result = {
                        "index": index,
                        "prompt": row["messages"][1]["content"],
                        "gold": row["messages"][2]["content"],
                        "prediction": prediction,
                        "match": evaluate_prediction(prediction, row, dataset_path.name),
                        "error": None,
                    }
                except RateLimitExceeded as error:
                    stopped_reason = f"rate_limited: {error}"
                    break
                except Exception as error:  # noqa: BLE001
                    result = {
                        "index": index,
                        "prompt": row["messages"][1]["content"],
                        "gold": row["messages"][2]["content"],
                        "prediction": "",
                        "match": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                append_result(sink, result)

            completed += 1
            correct += int(bool(result.get("match")))
            if result.get("error"):
                failed += 1
            if len(samples) < 5:
                samples.append(result)

    return {
        "dataset": str(dataset_path),
        "model": args.model,
        "count": completed,
        "correct": correct,
        "accuracy": (correct / completed) if completed else 0.0,
        "failed": failed,
        "stopped_reason": stopped_reason,
        "output_file": str(output_path),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenRouter model on llm-jepa datasets.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-strict-answer-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(benchmark(parse_args()), indent=2))


if __name__ == "__main__":
    main()
