from pathlib import Path
import sqlite3

from text_jepa.benchmarking import (
    benchmark_messages,
    dataset_task_name,
    gsm8k_final_answer,
    hellaswag_target_choice,
    score_prediction,
    summarize_distribution,
)


def make_row(user_text, assistant_text):
    return {
        "messages": [
            {"role": "system", "content": "Follow the task."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def test_dataset_task_name_strips_split_suffix():
    assert dataset_task_name("gsm8k_test.jsonl") == "gsm8k"
    assert dataset_task_name("spider_train.jsonl") == "spider"
    assert dataset_task_name("custom.jsonl") == "custom"


def test_benchmark_messages_adds_answer_only_instructions():
    messages = make_row("question", "answer")["messages"]
    prompt_messages = benchmark_messages(messages, dataset_name="synth_test.jsonl", strict_answer_only=True)

    assert len(prompt_messages) == 2
    assert "Return only the final answer." in prompt_messages[0]["content"]
    assert "Use the dataset's regex notation." in prompt_messages[0]["content"]


def test_benchmark_messages_adds_gsm8k_format_instruction():
    messages = make_row("question", "#### 18")["messages"]
    prompt_messages = benchmark_messages(messages, dataset_name="gsm8k_test.jsonl", strict_answer_only=True)

    assert "For GSM8K, return the final numeric answer exactly as `#### <answer>`." in prompt_messages[0]["content"]


def test_benchmark_messages_can_leave_prompt_unmodified():
    messages = make_row("question", "answer")["messages"]
    prompt_messages = benchmark_messages(messages, dataset_name="gsm8k_test.jsonl", strict_answer_only=False)

    assert prompt_messages == messages[:-1]


def test_gsm8k_final_answer_extracts_final_hash_answer():
    assert gsm8k_final_answer("work\n#### 18") == "18"
    assert gsm8k_final_answer("#### 18") == "18"
    assert gsm8k_final_answer("#### <answer> 18</answer>") == "18"
    assert gsm8k_final_answer("The final answer is 18.") == "18"
    assert gsm8k_final_answer("Answer: $1,234") == "1234"
    assert gsm8k_final_answer("no final marker") is None


def test_hellaswag_target_choice_normalizes_embedded_choice():
    assert hellaswag_target_choice("A") == "A"
    assert hellaswag_target_choice("The answer is C.") == "C"
    assert hellaswag_target_choice("not a choice") is None


def test_score_prediction_exact_match_task():
    row = make_row("sentiment", "Good")
    assert score_prediction("Good", row, "rotten_tomatoes_test.jsonl")
    assert not score_prediction("Bad", row, "rotten_tomatoes_test.jsonl")


def test_score_prediction_gsm8k_uses_final_answer():
    row = make_row("math", "steps\n#### 18")
    assert score_prediction("other steps\n#### 18", row, "gsm8k_test.jsonl")
    assert not score_prediction("other steps\n#### 19", row, "gsm8k_test.jsonl")


def test_score_prediction_nq_open_accepts_semicolon_separated_answers():
    row = make_row("qa", "Vancouver, British Columbia")
    assert score_prediction("Toronto; Vancouver", row, "nq_open_test.jsonl")
    assert not score_prediction("Toronto; Montreal", row, "nq_open_test.jsonl")


def test_score_prediction_hellaswag_accepts_choice_text():
    row = make_row("mcq", "C")
    assert score_prediction("The best answer is C.", row, "hellaswag_test.jsonl")
    assert not score_prediction("B", row, "hellaswag_test.jsonl")


def test_score_prediction_spider_executes_sql(tmp_path):
    db_root = tmp_path / "database"
    db_dir = db_root / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"

    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE city(name TEXT)")
        connection.execute("INSERT INTO city(name) VALUES ('a'), ('b')")
        connection.commit()

    row = {
        "messages": [
            {"role": "system", "content": "Convert natural language to SQL."},
            {"role": "user", "content": "For db_id:[toy]\n\nHow many cities are there?"},
            {"role": "assistant", "content": "SELECT count(*) FROM city"},
        ]
    }

    assert score_prediction("SELECT count(*) FROM city", row, "spider_test.jsonl", spider_path=db_root)
    assert not score_prediction("SELECT name FROM city", row, "spider_test.jsonl", spider_path=db_root)


def test_summarize_distribution_returns_expected_stats():
    summary = summarize_distribution([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "count": 4,
        "mean": 2.5,
        "std": (1.25**0.5),
        "p10": 1.3,
        "p20": 1.6,
        "p50": 2.5,
        "p80": 3.4000000000000004,
        "p90": 3.7,
    }


def test_summarize_distribution_returns_none_for_empty_input():
    assert summarize_distribution([]) is None
