from __future__ import annotations

import json
import re
import sqlite3
from math import sqrt
from pathlib import Path


ANSWER_ONLY = (
    "Return only the final answer. Do not explain. Do not use markdown. "
    "Do not include code fences. Do not include any extra text."
)
GSM8K_STYLE = "For GSM8K, return the final numeric answer exactly as `#### <answer>`."
SYNTH_STYLE = (
    "Use the dataset's regex notation. Use ~ for negation and & for conjunction when needed. "
    "Do not add anchors like ^ or $. Return only the regex expression."
)
GSM8K_FINAL_ANSWER = re.compile(r"(?:^|\n)####\s*(.+)$")
GSM8K_FALLBACK_NUMBER = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?")
SPIDER_DB_ID = re.compile(r"For db_id:\[(.+)\]")
HELLASWAG_CHOICE = re.compile(r"\b([ABCD])\b")


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


def append_result(handle, result: dict) -> None:
    handle.write(json.dumps(result) + "\n")
    handle.flush()


def summarize_distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = [float(value) for value in values]
    ordered.sort()
    count = len(ordered)

    mean = 0.0
    sum_squared_deltas = 0.0
    for index, value in enumerate(ordered, start=1):
        delta = value - mean
        mean += delta / index
        sum_squared_deltas += delta * (value - mean)
    variance = sum_squared_deltas / count

    def percentile(fraction: float) -> float:
        if count == 1:
            return ordered[0]
        position = fraction * (count - 1)
        lower = int(position)
        upper = min(lower + 1, count - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": count,
        "mean": mean,
        "std": sqrt(variance),
        "p10": percentile(0.10),
        "p20": percentile(0.20),
        "p50": percentile(0.50),
        "p80": percentile(0.80),
        "p90": percentile(0.90),
    }


def dataset_task_name(dataset_name: str) -> str:
    stem = Path(dataset_name).stem
    for suffix in ("_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_synth_like_dataset(dataset_name: str) -> bool:
    task_name = dataset_task_name(dataset_name)
    return task_name == "synth" or task_name.endswith("_synth")


def benchmark_messages(
    messages: list[dict[str, str]],
    *,
    dataset_name: str,
    strict_answer_only: bool,
) -> list[dict[str, str]]:
    prompt_messages = [dict(message) for message in messages[:-1]]
    if not strict_answer_only:
        return prompt_messages

    instructions = [ANSWER_ONLY]
    if dataset_task_name(dataset_name) == "gsm8k":
        instructions.append(GSM8K_STYLE)
    if is_synth_like_dataset(dataset_name):
        instructions.append(SYNTH_STYLE)

    if prompt_messages and prompt_messages[0]["role"] == "system":
        prompt_messages[0]["content"] = prompt_messages[0]["content"].rstrip() + "\n\n" + "\n".join(instructions)
    else:
        prompt_messages.insert(0, {"role": "system", "content": "\n".join(instructions)})
    return prompt_messages


def prompt_text(row: dict) -> str:
    messages = row["messages"]
    if len(messages) < 2:
        return ""
    return messages[-2]["content"]


def gold_text(row: dict) -> str:
    return row["messages"][-1]["content"]


def hellaswag_target_choice(text: str) -> str | None:
    stripped = text.strip()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    match = HELLASWAG_CHOICE.search(stripped)
    return None if match is None else match.group(1)


def gsm8k_final_answer(text: str) -> str | None:
    def extract_last_number(candidate: str) -> str | None:
        matches = GSM8K_FALLBACK_NUMBER.findall(candidate)
        if not matches:
            return None
        return matches[-1].replace("$", "").replace(",", "").strip()

    match = re.search(GSM8K_FINAL_ANSWER, text)
    if match is not None:
        extracted = extract_last_number(match.group(1))
        if extracted is not None:
            return extracted
        return match.group(1).strip()

    return extract_last_number(text)


def spider_db_path(messages: list[dict[str, str]], spider_path: str | Path) -> Path:
    if not spider_path:
        raise ValueError("spider_path is required for Spider evaluation")
    db_match = re.search(SPIDER_DB_ID, messages[-2]["content"])
    if db_match is None:
        raise ValueError("Could not extract db_id from Spider example")
    db_id = db_match.group(1)
    return Path(spider_path) / db_id / f"{db_id}.sqlite"


def execute_sql(db_path: Path, query: str):
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(query)
        return cursor.fetchall()


def score_prediction(
    prediction: str,
    row: dict,
    dataset_name: str,
    *,
    spider_path: str | Path = "",
    startswith: bool = False,
) -> bool:
    task_name = dataset_task_name(dataset_name)
    target = gold_text(row)

    if startswith:
        return prediction.startswith(target)

    if task_name == "gsm8k":
        return gsm8k_final_answer(prediction) == gsm8k_final_answer(target)

    if task_name == "hellaswag":
        return hellaswag_target_choice(prediction) == hellaswag_target_choice(target)

    if task_name == "spider":
        db_path = spider_db_path(row["messages"], spider_path)
        try:
            generated_result = execute_sql(db_path, prediction)
            gold_result = execute_sql(db_path, target)
        except sqlite3.Error:
            return False
        return generated_result == gold_result

    if task_name == "nq_open":
        return any(answer in target for answer in prediction.split("; "))

    return prediction == target
