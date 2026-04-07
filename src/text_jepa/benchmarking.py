from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


ANSWER_ONLY = (
    "Return only the final answer. Do not explain. Do not use markdown. "
    "Do not include code fences. Do not include any extra text."
)
SYNTH_STYLE = (
    "Use the dataset's regex notation. Use ~ for negation and & for conjunction when needed. "
    "Do not add anchors like ^ or $. Return only the regex expression."
)
GSM8K_FINAL_ANSWER = re.compile(r"\n#### (.+)$")
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


def dataset_task_name(dataset_name: str) -> str:
    stem = Path(dataset_name).stem
    for suffix in ("_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


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
    if dataset_task_name(dataset_name) == "synth":
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
    match = re.search(GSM8K_FINAL_ANSWER, text)
    return None if match is None else match.group(1)


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

