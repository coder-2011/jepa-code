set shell := ["zsh", "-lc"]

venv := ".venv"

test-batching:
    source {{venv}}/bin/activate && pytest tests/test_batching.py

test-attention-masks:
    source {{venv}}/bin/activate && pytest tests/test_attention_masks.py

test-benchmarking:
    source {{venv}}/bin/activate && pytest tests/test_benchmarking.py

test-embeddings:
    source {{venv}}/bin/activate && pytest tests/test_embeddings.py

test-repro:
    source {{venv}}/bin/activate && pytest tests/test_repro.py

test-fineweb-dataloader:
    source {{venv}}/bin/activate && pytest tests/test_fineweb_dataloader.py

test-finetune-gsm8k-hf:
    source {{venv}}/bin/activate && pytest tests/test_finetune_gsm8k_hf.py

test-llm-jepa:
    source {{venv}}/bin/activate && pytest tests/test_llm_jepa.py

test-masking:
    source {{venv}}/bin/activate && pytest tests/test_masking.py

test-tokenization:
    source {{venv}}/bin/activate && pytest tests/test_tokenization.py

test-stp-objective:
    source {{venv}}/bin/activate && pytest tests/test_stp_objective.py

test-train-checkpointing:
    source {{venv}}/bin/activate && pytest tests/test_train_checkpointing.py

test-train-llm-jepa-script:
    source {{venv}}/bin/activate && pytest tests/test_train_llm_jepa_script.py

all-tests:
    source {{venv}}/bin/activate && pytest tests

download-fineweb-sample max_bytes="5242880" name="CC-MAIN-2024-10" output="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/download_fineweb_sample.py --name {{name}} --max-bytes {{max_bytes}} --output {{output}}

train-layer steps="50" batch_size="2" data_path="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/train.py --steps {{steps}} --batch-size {{batch_size}} --data-path {{data_path}}

train-llm-jepa steps="10" batch_size="1" max_length="256" model_name="hf-internal-testing/tiny-random-gpt2" train_file="llm-jepa/datasets/synth_train.jsonl" eval_file="llm-jepa/datasets/synth_test.jsonl":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --steps {{steps}} --batch-size {{batch_size}} --max-length {{max_length}} --model-name {{model_name}} --save-every 0 --train-file {{train_file}} --eval-file {{eval_file}}

train-llm-jepa-qwen steps="1" batch_size="1" max_length="64" max_docs="1" device="cpu" train_file="llm-jepa/datasets/synth_train.jsonl" eval_file="llm-jepa/datasets/synth_test.jsonl" checkpoint_dir="checkpoints/llm-jepa-qwen":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --steps {{steps}} --batch-size {{batch_size}} --max-length {{max_length}} --max-docs {{max_docs}} --device {{device}} --model-name Qwen/Qwen3-0.6B --checkpoint-dir {{checkpoint_dir}} --save-every {{steps}} --train-file {{train_file}} --eval-file {{eval_file}} --wandb-mode offline

benchmark-openrouter dataset="llm-jepa/datasets/synth_test.jsonl" model="qwen/qwen3.5-9b" judge_model="openai/gpt-5.4" max_examples="10" output="tmp/openrouter-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_openrouter.py --dataset {{dataset}} --model {{model}} --judge-model {{judge_model}} --max-examples {{max_examples}} --output {{output}} --force

benchmark-local-qwen dataset="llm-jepa/datasets/synth_test.jsonl" base_model="Qwen/Qwen3-0.6B" checkpoint="path/to/qwen-0.6b-checkpoint.pt" judge_model="openai/gpt-5.4" max_examples="10" device="cuda" output="tmp/local-qwen-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model {{base_model}} --checkpoint {{checkpoint}} --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force

benchmark-local-qwen-base dataset="llm-jepa/datasets/synth_test.jsonl" judge_model="openai/gpt-5.4" max_examples="10" device="cuda" output="tmp/local-qwen-base-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint '' --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force

benchmark-local-qwen-tuned dataset="llm-jepa/datasets/synth_test.jsonl" checkpoint="checkpoints/llm-jepa-qwen/latest.pt" judge_model="openai/gpt-5.4" max_examples="10" device="cuda" output="tmp/local-qwen-tuned-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint {{checkpoint}} --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force

benchmark-local-qwen-compare dataset="llm-jepa/datasets/synth_test.jsonl" checkpoint="checkpoints/llm-jepa-qwen/latest.pt" judge_model="openai/gpt-5.4" max_examples="10" device="cuda" base_output="tmp/local-qwen-base-benchmark.jsonl" tuned_output="tmp/local-qwen-tuned-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint '' --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{base_output}} --force
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint {{checkpoint}} --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{tuned_output}} --force

prepare-gsm8k train_output="tmp/gsm8k/gsm8k_train.jsonl" test_output="tmp/gsm8k/gsm8k_test.jsonl":
    source {{venv}}/bin/activate && mkdir -p tmp/gsm8k && python -c 'import json; from pathlib import Path; from datasets import load_dataset; system = "Solve the math word problem carefully. Show the reasoning and end with the final answer on its own line as #### <number>."; outputs = {"train": Path("{{train_output}}"), "test": Path("{{test_output}}")}; \
for split, path in outputs.items(): \
    path.parent.mkdir(parents=True, exist_ok=True); dataset = load_dataset("gsm8k", "main", split=split); \
    handle = path.open("w", encoding="utf-8"); \
    [handle.write(json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": row["question"].strip()}, {"role": "assistant", "content": row["answer"].strip()}], "text": [{"role": "system", "content": system}, {"role": "user", "content": row["question"].strip()}], "code": [{"role": "assistant", "content": row["answer"].strip()}]}, ensure_ascii=False) + "\n") for row in dataset]; \
    handle.close(); print(split, len(dataset), path)'

train-gsm8k-sft-hf steps="200" batch_size="32" max_length="1024" max_docs="1024" eval_max_docs="256" device="cuda" output_dir="tmp/runs/gsm8k_sft_qwen3_0_6b" train_file="tmp/gsm8k/gsm8k_train.jsonl" eval_file="tmp/gsm8k/gsm8k_test.jsonl":
    source {{venv}}/bin/activate && python scripts/finetune_gsm8k_hf.py --model-name Qwen/Qwen3-0.6B --train-file {{train_file}} --eval-file {{eval_file}} --output-dir {{output_dir}} --steps {{steps}} --batch-size {{batch_size}} --grad-accumulation 1 --max-docs {{max_docs}} --eval-max-docs {{eval_max_docs}} --max-length {{max_length}} --device {{device}} --dtype bfloat16 --num-workers 4 --pin-memory --tf32 --wandb-mode offline

train-gsm8k-llm-jepa-fast steps="200" batch_size="16" max_length="1024" max_docs="1024" device="cuda" checkpoint_dir="tmp/runs/llm_jepa_gsm8k_qwen3_0_6b" train_file="tmp/gsm8k/gsm8k_train.jsonl" eval_file="tmp/gsm8k/gsm8k_test.jsonl":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --model-name Qwen/Qwen3-0.6B --train-file {{train_file}} --eval-file {{eval_file}} --steps {{steps}} --batch-size {{batch_size}} --max-docs {{max_docs}} --max-length {{max_length}} --device {{device}} --dtype bfloat16 --num-workers 4 --pin-memory --tf32 --checkpoint-dir {{checkpoint_dir}} --save-every 100 --val-every 50 --val-max-batches 8 --wandb-mode offline

benchmark-gsm8k-base max_examples="200" max_length="1024" max_new_tokens="256" device="cuda" dataset="tmp/gsm8k/gsm8k_test.jsonl" output="tmp/bench_base_200.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint '' --max-examples {{max_examples}} --max-length {{max_length}} --max-new-tokens {{max_new_tokens}} --device {{device}} --output {{output}} --force

benchmark-gsm8k-sft max_examples="200" max_length="1024" max_new_tokens="256" device="cuda" dataset="tmp/gsm8k/gsm8k_test.jsonl" base_model="tmp/runs/gsm8k_sft_qwen3_0_6b/merged" output="tmp/bench_sft_200.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model {{base_model}} --checkpoint '' --max-examples {{max_examples}} --max-length {{max_length}} --max-new-tokens {{max_new_tokens}} --device {{device}} --output {{output}} --force

benchmark-gsm8k-llm-jepa max_examples="200" max_length="1024" max_new_tokens="256" device="cuda" dataset="tmp/gsm8k/gsm8k_test.jsonl" checkpoint="tmp/runs/llm_jepa_gsm8k_qwen3_0_6b/step-000200.pt" output="tmp/bench_jepa_200.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint {{checkpoint}} --max-examples {{max_examples}} --max-length {{max_length}} --max-new-tokens {{max_new_tokens}} --device {{device}} --output {{output}} --force

benchmark-gsm8k-base-verbose max_examples="10" max_length="1024" max_new_tokens="256" device="cuda" dataset="tmp/gsm8k/gsm8k_test.jsonl" output="tmp/bench_base_verbose.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint '' --max-examples {{max_examples}} --max-length {{max_length}} --max-new-tokens {{max_new_tokens}} --device {{device}} --output {{output}} --force --verbose
